"""
Experiment Manager for AXIOM Extensions

Manages multiple model experiments, comparisons, and automated testing
for AXIOM architecture extensions.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .model_runner import ModelRunner, RunConfig, RunResult

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    description: str
    model_configs: List[Dict[str, Any]]
    game_names: List[str]
    num_runs_per_config: int = 3
    run_config_template: RunConfig = None
    parallel_execution: bool = False
    max_workers: int = 4

    def __post_init__(self):
        if self.run_config_template is None:
            self.run_config_template = RunConfig(
                model_type="default",
                game_name="default",
                num_steps=1000,
                num_episodes=5
            )


@dataclass
class ExperimentResult:
    """Results from an experiment."""
    experiment_name: str
    total_runs: int
    successful_runs: int
    failed_runs: int
    run_results: List[RunResult]
    best_run: Optional[RunResult]
    worst_run: Optional[RunResult]
    average_performance: Dict[str, float]
    duration_seconds: float
    start_time: float
    end_time: float


class ExperimentManager:
    """
    Manages experiments involving multiple model configurations and games.

    This class provides methods to:
    - Run multiple experiments with different configurations
    - Compare performance across configurations
    - Generate experiment reports
    - Manage experiment metadata
    """

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        """
        Initialize experiment manager.

        Args:
            base_dir: Base directory for experiments (if None, uses output config)
        """
        # Import output configuration
        try:
            from ..output_config import output_config
            if base_dir is None:
                # Use centralized output configuration
                self.base_dir = output_config.get_path_for("experiments")
            else:
                self.base_dir = Path(base_dir)
        except ImportError:
            # Fallback to default behavior if output_config is not available
            self.base_dir = Path(base_dir) if base_dir else Path.cwd() / "experiments"

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Initialize model runner
        try:
            from ..output_config import output_config
            self.model_runner = ModelRunner(output_config.get_path_for("runs"))
        except ImportError:
            # Fallback to default behavior if output_config is not available
            self.model_runner = ModelRunner(self.base_dir.parent / "runs")

    def run_experiment(
        self,
        config: ExperimentConfig,
        model_factory: Callable[[Dict[str, Any]], Any],
        game_factory: Callable[[str], Any],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> ExperimentResult:
        """
        Run a complete experiment with multiple configurations.

        Args:
            config: Experiment configuration
            model_factory: Function to create models
            game_factory: Function to create games
            progress_callback: Optional progress callback

        Returns:
            ExperimentResult: Results of the experiment
        """
        start_time = time.time()
        self.logger.info(f"Starting experiment: {config.name}")

        all_run_results = []

        if config.parallel_execution:
            results = self._run_parallel_experiment(
                config, model_factory, game_factory, progress_callback
            )
            all_run_results.extend(results)
        else:
            results = self._run_sequential_experiment(
                config, model_factory, game_factory, progress_callback
            )
            all_run_results.extend(results)

        # Analyze results
        successful_runs = [r for r in all_run_results if r.success]
        failed_runs = [r for r in all_run_results if not r.success]

        best_run = max(successful_runs, key=lambda r: r.average_reward) if successful_runs else None
        worst_run = min(successful_runs, key=lambda r: r.average_reward) if successful_runs else None

        average_performance = self._calculate_average_performance(successful_runs)

        experiment_result = ExperimentResult(
            experiment_name=config.name,
            total_runs=len(all_run_results),
            successful_runs=len(successful_runs),
            failed_runs=len(failed_runs),
            run_results=all_run_results,
            best_run=best_run,
            worst_run=worst_run,
            average_performance=average_performance,
            duration_seconds=time.time() - start_time,
            start_time=start_time,
            end_time=time.time()
        )

        # Save experiment results
        self.save_experiment_result(experiment_result)

        self.logger.info(f"Experiment {config.name} completed in {experiment_result.duration_seconds:.2f}s")
        self.logger.info(f"Results: {len(successful_runs)}/{len(all_run_results)} runs successful")

        return experiment_result

    def _run_sequential_experiment(
        self,
        config: ExperimentConfig,
        model_factory: Callable,
        game_factory: Callable,
        progress_callback: Optional[Callable] = None
    ) -> List[RunResult]:
        """Run experiment sequentially."""
        results = []

        total_combinations = len(config.model_configs) * len(config.game_names) * config.num_runs_per_config
        current_run = 0

        for model_config in config.model_configs:
            for game_name in config.game_names:
                for run_idx in range(config.num_runs_per_config):
                    # Create run configuration
                    run_config = RunConfig(
                        model_type=model_config.get("model_type", "unknown"),
                        game_name=game_name,
                        num_steps=config.run_config_template.num_steps,
                        num_episodes=config.run_config_template.num_episodes,
                        model_config=model_config
                    )

                    # Update progress
                    current_run += 1
                    if progress_callback:
                        progress_callback({
                            "experiment": config.name,
                            "current_run": current_run,
                            "total_runs": total_combinations,
                            "model_config": model_config,
                            "game_name": game_name,
                            "run_idx": run_idx
                        })

                    # Execute run
                    result = self.model_runner.run_model(
                        run_config, model_factory, game_factory
                    )
                    results.append(result)

        return results

    def _run_parallel_experiment(
        self,
        config: ExperimentConfig,
        model_factory: Callable,
        game_factory: Callable,
        progress_callback: Optional[Callable] = None
    ) -> List[RunResult]:
        """Run experiment in parallel."""
        results = []

        def run_single_experiment(model_config, game_name, run_idx):
            run_config = RunConfig(
                model_type=model_config.get("model_type", "unknown"),
                game_name=game_name,
                num_steps=config.run_config_template.num_steps,
                num_episodes=config.run_config_template.num_episodes,
                model_config=model_config
            )

            return self.model_runner.run_model(run_config, model_factory, game_factory)

        # Create all experiment combinations
        experiment_params = []
        for model_config in config.model_configs:
            for game_name in config.game_names:
                for run_idx in range(config.num_runs_per_config):
                    experiment_params.append((model_config, game_name, run_idx))

        # Execute in parallel
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            future_to_params = {
                executor.submit(run_single_experiment, *params): params
                for params in experiment_params
            }

            for future in as_completed(future_to_params):
                result = future.result()
                results.append(result)

                if progress_callback:
                    params = future_to_params[future]
                    progress_callback({
                        "experiment": config.name,
                        "completed_runs": len(results),
                        "total_runs": len(experiment_params),
                        "current_params": params
                    })

        return results

    def _calculate_average_performance(self, successful_runs: List[RunResult]) -> Dict[str, float]:
        """Calculate average performance across successful runs."""
        if not successful_runs:
            return {
                "average_reward": 0.0,
                "best_reward": 0.0,
                "worst_reward": 0.0,
                "total_reward": 0.0,
                "average_duration": 0.0
            }

        if not successful_runs:
            return {
                "average_reward": 0.0,
                "best_reward": 0.0,
                "worst_reward": 0.0,
                "total_reward": 0.0,
                "average_duration": 0.0
            }

        return {
            "average_reward": sum(r.average_reward for r in successful_runs) / len(successful_runs),
            "best_reward": max(r.best_reward for r in successful_runs),
            "worst_reward": min(r.worst_reward for r in successful_runs),
            "total_reward": sum(r.total_reward for r in successful_runs),
            "average_duration": sum(r.duration_seconds for r in successful_runs) / len(successful_runs)
        }

    def compare_experiments(self, experiment_names: List[str]) -> Dict[str, Any]:
        """
        Compare multiple experiments.

        Args:
            experiment_names: Names of experiments to compare

        Returns:
            Dictionary with comparison results
        """
        experiments = []
        for name in experiment_names:
            result = self.load_experiment_result(name)
            if result:
                experiments.append(result)

        if not experiments:
            return {"error": "No experiments found"}

        comparison = {
            "experiments_compared": len(experiments),
            "experiment_names": experiment_names,
            "best_experiment": max(experiments, key=lambda e: e.average_performance["average_reward"]),
            "worst_experiment": min(experiments, key=lambda e: e.average_performance["average_reward"]),
            "performance_ranking": sorted(
                experiments,
                key=lambda e: e.average_performance["average_reward"],
                reverse=True
            ),
            "summary": {
                "total_runs": sum(e.total_runs for e in experiments),
                "total_successful": sum(e.successful_runs for e in experiments),
                "total_failed": sum(e.failed_runs for e in experiments),
                "average_performance": {
                    "average_reward": sum(e.average_performance["average_reward"] for e in experiments) / len(experiments),
                    "best_reward": max(e.average_performance["best_reward"] for e in experiments),
                    "total_duration": sum(e.duration_seconds for e in experiments)
                }
            }
        }

        return comparison

    def list_experiments(self) -> List[str]:
        """List all available experiments."""
        if not self.base_dir.exists():
            return []

        return [d.name for d in self.base_dir.iterdir() if d.is_dir()]

    def load_experiment_result(self, experiment_name: str) -> Optional[ExperimentResult]:
        """Load an experiment result from disk."""
        exp_dir = self.base_dir / experiment_name
        result_file = exp_dir / "experiment_result.json"

        if not result_file.exists():
            return None

        try:
            import json
            with open(result_file, 'r') as f:
                data = json.load(f)

            # Convert run_results back to RunResult objects
            run_results = [RunResult(**run_data) for run_data in data["run_results"]]

            return ExperimentResult(
                experiment_name=data["experiment_name"],
                total_runs=data["total_runs"],
                successful_runs=data["successful_runs"],
                failed_runs=data["failed_runs"],
                run_results=run_results,
                best_run=RunResult(**data["best_run"]) if data["best_run"] else None,
                worst_run=RunResult(**data["worst_run"]) if data["worst_run"] else None,
                average_performance=data["average_performance"],
                duration_seconds=data["duration_seconds"],
                start_time=data["start_time"],
                end_time=data["end_time"]
            )
        except Exception as e:
            self.logger.error(f"Failed to load experiment {experiment_name}: {e}")
            return None

    def save_experiment_result(self, result: ExperimentResult) -> None:
        """Save an experiment result to disk."""
        exp_dir = self.base_dir / result.experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        result_file = exp_dir / "experiment_result.json"

        try:
            import json

            # Convert to serializable format
            data = {
                "experiment_name": result.experiment_name,
                "total_runs": result.total_runs,
                "successful_runs": result.successful_runs,
                "failed_runs": result.failed_runs,
                "run_results": [asdict(run) for run in result.run_results],
                "best_run": asdict(result.best_run) if result.best_run else None,
                "worst_run": asdict(result.worst_run) if result.worst_run else None,
                "average_performance": result.average_performance,
                "duration_seconds": result.duration_seconds,
                "start_time": result.start_time,
                "end_time": result.end_time
            }

            with open(result_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save experiment {result.experiment_name}: {e}")

    def generate_experiment_report(self, experiment_name: str) -> str:
        """
        Generate a comprehensive report for an experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Formatted experiment report
        """
        result = self.load_experiment_result(experiment_name)

        if not result:
            return f"Experiment '{experiment_name}' not found"

        report = ".2f"".2f"".2f"f"""
EXPERIMENT REPORT: {experiment_name}
{'=' * (18 + len(experiment_name))}

Overview:
- Total Runs: {result.total_runs}
- Successful Runs: {result.successful_runs}
- Failed Runs: {result.failed_runs}
- Success Rate: {result.successful_runs / result.total_runs * 100:.1f}%
- Duration: {result.duration_seconds:.2f}s

Performance Summary:
- Average Reward: {result.average_performance['average_reward']:.2f}
- Best Reward: {result.average_performance['best_reward']:.2f}
- Worst Reward: {result.average_performance['worst_reward']:.2f}
- Total Reward: {result.average_performance['total_reward']:.2f}

Best Run:
{self._format_run_result(result.best_run) if result.best_run else 'No successful runs'}

Configuration Summary:
{self._generate_configuration_summary(result)}
"""

        return report.strip()

    def _format_run_result(self, run_result: RunResult) -> str:
        """Format a run result for display."""
        return ".2f"".2f"".2f"".2f"f"""
- Run ID: {run_result.run_id}
- Model Type: {run_result.model_type}
- Game: {run_result.game_name}
- Steps: {run_result.total_steps}
- Episodes: {run_result.total_episodes}
- Average Reward: {run_result.average_reward:.2f}
- Best Reward: {run_result.best_reward:.2f}
- Duration: {run_result.duration_seconds:.2f}s"""

    def _generate_configuration_summary(self, result: ExperimentResult) -> str:
        """Generate configuration summary from run results."""
        if not result.run_results:
            return "No run results available"

        # Group by model type and game
        config_summary = {}
        for run in result.run_results:
            key = f"{run.model_type}_{run.game_name}"
            if key not in config_summary:
                config_summary[key] = []
            config_summary[key].append(run)

        summary_lines = []
        for config_key, runs in config_summary.items():
            successful_runs = [r for r in runs if r.success]
            if successful_runs:
                avg_reward = sum(r.average_reward for r in successful_runs) / len(successful_runs)
                summary_lines.append(f"- {config_key}: {len(successful_runs)}/{len(runs)} runs, avg reward: {avg_reward:.2f}")
            else:
                summary_lines.append(f"- {config_key}: {len(successful_runs)}/{len(runs)} runs (all failed)")

        return "\n".join(summary_lines)
