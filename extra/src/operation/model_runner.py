"""
Model Runner for AXIOM Extensions

Provides comprehensive model execution and management capabilities
for AXIOM architecture extensions.
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Configuration for model execution."""
    model_type: str
    game_name: str
    num_steps: int = 10000
    num_episodes: int = 10
    save_frequency: int = 1000
    log_frequency: int = 100
    render_frequency: Optional[int] = None
    random_seed: Optional[int] = None
    output_dir: str = "results"
    checkpoint_dir: str = "checkpoints"
    video_output: bool = True
    csv_output: bool = True
    wandb_logging: bool = False
    model_config: Dict[str, Any] = None

    def __post_init__(self):
        if self.model_config is None:
            self.model_config = {}


@dataclass
class RunResult:
    """Results from a model execution run."""
    run_id: str
    model_type: str
    game_name: str
    total_steps: int
    total_episodes: int
    total_reward: float
    average_reward: float
    best_reward: float
    worst_reward: float
    duration_seconds: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ModelRunner:
    """
    Executes and manages AXIOM model runs with comprehensive monitoring
    and result tracking capabilities.
    """

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        """
        Initialize model runner.

        Args:
            base_dir: Base directory for runs and results (if None, uses output config)
        """
        # Import output configuration
        try:
            from ..output_config import output_config
            if base_dir is None:
                # Use centralized output configuration
                self.base_dir = output_config.get_path_for("runs")
            else:
                self.base_dir = Path(base_dir)
        except ImportError:
            # Fallback to default behavior if output_config is not available
            self.base_dir = Path(base_dir) if base_dir else Path.cwd() / "runs"

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self._current_run: Optional[RunResult] = None

    def run_model(
        self,
        config: RunConfig,
        model_factory: Callable[[Dict[str, Any]], Any],
        game_factory: Callable[[str], Any],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> RunResult:
        """
        Execute a model run with the given configuration.

        Args:
            config: Run configuration
            model_factory: Function that creates model from config
            game_factory: Function that creates game environment
            progress_callback: Optional callback for progress updates

        Returns:
            RunResult: Results of the model execution
        """
        run_id = self._generate_run_id()
        start_time = time.time()

        self.logger.info(f"Starting run {run_id} for {config.model_type} on {config.game_name}")

        # Create output directories
        run_dir = self.base_dir / run_id
        output_dir = run_dir / config.output_dir
        checkpoint_dir = run_dir / config.checkpoint_dir

        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Initialize model and game
            model = model_factory(config.model_config)
            game = game_factory(config.game_name)

            # Set random seed if specified
            if config.random_seed is not None:
                self._set_random_seed(config.random_seed)

            # Execute the run
            result = self._execute_run(
                config, model, game, output_dir, checkpoint_dir, progress_callback
            )

            # Create final result
            run_result = RunResult(
                run_id=run_id,
                model_type=config.model_type,
                game_name=config.game_name,
                total_steps=result["total_steps"],
                total_episodes=result["total_episodes"],
                total_reward=result["total_reward"],
                average_reward=result["average_reward"],
                best_reward=result["best_reward"],
                worst_reward=result["worst_reward"],
                duration_seconds=time.time() - start_time,
                success=True,
                metadata={
                    "run_dir": str(run_dir),
                    "output_dir": str(output_dir),
                    "checkpoint_dir": str(checkpoint_dir),
                    "config": asdict(config)
                }
            )

            self._current_run = run_result
            self.logger.info(f"Run {run_id} completed successfully")

            return run_result

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Run {run_id} failed: {error_msg}")

            run_result = RunResult(
                run_id=run_id,
                model_type=config.model_type,
                game_name=config.game_name,
                total_steps=0,
                total_episodes=0,
                total_reward=0.0,
                average_reward=0.0,
                best_reward=0.0,
                worst_reward=0.0,
                duration_seconds=time.time() - start_time,
                success=False,
                error_message=error_msg
            )

            return run_result

    def _execute_run(
        self,
        config: RunConfig,
        model: Any,
        game: Any,
        output_dir: Path,
        checkpoint_dir: Path,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Execute the actual model run."""
        total_steps = 0
        total_episodes = 0
        episode_rewards = []
        all_rewards = []

        step = 0
        episode = 0

        while step < config.num_steps and episode < config.num_episodes:
            episode_reward = 0
            episode_steps = 0

            # Reset game for new episode
            game.reset()

            while not game.is_terminal() and step < config.num_steps:
                # Get model action
                state = game.get_state()
                action = model.act(state)

                # Execute action
                reward, next_state, terminal = game.step(action)

                # Train model
                model.train(state, action, reward, next_state, terminal)

                episode_reward += reward
                all_rewards.append(reward)
                step += 1
                episode_steps += 1

                # Periodic operations
                if step % config.save_frequency == 0:
                    self._save_checkpoint(model, checkpoint_dir, step)

                if step % config.log_frequency == 0:
                    self.logger.info(f"Step {step}: Episode {episode}, Reward: {episode_reward:.2f}")
                    if progress_callback:
                        progress_callback({
                            "step": step,
                            "episode": episode,
                            "episode_reward": episode_reward,
                            "total_reward": sum(all_rewards),
                            "average_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0
                        })

                if config.render_frequency and step % config.render_frequency == 0:
                    self._save_render(game, output_dir, step)

            # Episode completed
            episode_rewards.append(episode_reward)
            episode += 1
            total_episodes += 1

        # Calculate final statistics
        result = {
            "total_steps": step,
            "total_episodes": total_episodes,
            "total_reward": sum(all_rewards),
            "average_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0,
            "best_reward": max(episode_rewards) if episode_rewards else 0,
            "worst_reward": min(episode_rewards) if episode_rewards else 0,
            "episode_rewards": episode_rewards,
            "all_rewards": all_rewards
        }

        return result

    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        return f"run_{timestamp}"

    def _set_random_seed(self, seed: int):
        """Set random seed for reproducibility."""
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)

        # Set torch seed if available
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        except ImportError:
            pass

    def _save_checkpoint(self, model: Any, checkpoint_dir: Path, step: int):
        """Save model checkpoint."""
        try:
            checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
            if hasattr(model, 'save'):
                model.save(checkpoint_path)
            else:
                self.logger.warning(f"Model does not have save method, skipping checkpoint at step {step}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint at step {step}: {e}")

    def _save_render(self, game: Any, output_dir: Path, step: int):
        """Save game render/frame."""
        try:
            if hasattr(game, 'render'):
                render_path = output_dir / "02d"
                game.render(save_path=render_path)
        except Exception as e:
            self.logger.error(f"Failed to save render at step {step}: {e}")

    def get_current_run(self) -> Optional[RunResult]:
        """Get the current/last run result."""
        return self._current_run

    def list_runs(self) -> List[str]:
        """List all available runs."""
        if not self.base_dir.exists():
            return []

        return [d.name for d in self.base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]

    def load_run_result(self, run_id: str) -> Optional[RunResult]:
        """Load a run result from disk."""
        run_dir = self.base_dir / run_id
        result_file = run_dir / "result.json"

        if not result_file.exists():
            return None

        try:
            import json
            with open(result_file, 'r') as f:
                data = json.load(f)

            return RunResult(**data)
        except Exception as e:
            self.logger.error(f"Failed to load run result {run_id}: {e}")
            return None

    def save_run_result(self, result: RunResult) -> None:
        """Save a run result to disk."""
        run_dir = self.base_dir / result.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        result_file = run_dir / "result.json"

        try:
            import json
            with open(result_file, 'w') as f:
                json.dump(asdict(result), f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save run result {result.run_id}: {e}")

    @contextmanager
    def managed_run(self, config: RunConfig):
        """
        Context manager for managed model runs with automatic cleanup.

        Args:
            config: Run configuration

        Yields:
            Dictionary with run information
        """
        run_info = {
            "run_id": self._generate_run_id(),
            "config": config,
            "start_time": time.time(),
            "status": "initializing"
        }

        try:
            run_info["status"] = "running"
            yield run_info
        finally:
            run_info["end_time"] = time.time()
            run_info["duration"] = run_info["end_time"] - run_info["start_time"]
            run_info["status"] = "completed"

            self.logger.info(f"Managed run {run_info['run_id']} completed in {run_info['duration']:.2f}s")
