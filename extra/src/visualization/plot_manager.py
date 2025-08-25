"""
Plot Manager for AXIOM Extensions

Provides high-level plotting management and batch visualization tools
for AXIOM experimental results and analysis.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from pathlib import Path
from datetime import datetime
import json

from .visualizer import Visualizer, VisualizationError
from .animator import Animator, AnimationError

logger = logging.getLogger(__name__)


class PlotManager:
    """
    Manages visualization and animation workflows for AXIOM experiments.

    This class provides methods to:
    - Create comprehensive visualization suites
    - Manage batch plotting operations
    - Generate experiment reports with visualizations
    - Handle visualization configuration and themes
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize plot manager.

        Args:
            output_dir: Base directory for all visualizations (if None, uses output config)
        """
        # Import output configuration
        try:
            from ..output_config import output_config
            if output_dir is None:
                # Use centralized output configuration
                self.output_dir = output_config.get_path_for("visualizations")
            else:
                self.output_dir = Path(output_dir)
        except ImportError:
            # Fallback to default behavior if output_config is not available
            self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "plots"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Initialize components with specific subdirectories
        try:
            from ..output_config import output_config
            self.visualizer = Visualizer(output_config.get_path_for("visualizations", "static"), style="default")
            self.animator = Animator(output_config.get_path_for("visualizations", "animations"), fps=30)
        except ImportError:
            # Fallback to default behavior if output_config is not available
            self.visualizer = Visualizer(self.output_dir / "static", style="default")
            self.animator = Animator(self.output_dir / "animations", fps=30)

    def create_experiment_visualization_suite(
        self,
        experiment_data: Dict[str, Any],
        experiment_name: str,
        include_animations: bool = True,
        theme: str = "default"
    ) -> Dict[str, Any]:
        """
        Create a comprehensive visualization suite for an experiment.

        Args:
            experiment_data: Complete experiment data
            experiment_name: Name of the experiment
            include_animations: Whether to create animations
            theme: Visualization theme

        Returns:
            Dictionary with paths to created visualizations
        """
        suite_dir = self.output_dir / f"{experiment_name}_suite"
        suite_dir.mkdir(parents=True, exist_ok=True)

        # Set theme
        self.visualizer.set_style(theme)

        created_files = {
            "static_plots": [],
            "animations": [],
            "reports": [],
            "data_exports": []
        }

        try:
            # Extract data components
            run_results = experiment_data.get('run_results', [])
            training_history = experiment_data.get('training_history', [])
            comparison_data = experiment_data.get('comparison_data', {})

            # 1. Performance comparison plots
            if run_results:
                perf_plot = self._create_performance_comparison(run_results, suite_dir, experiment_name)
                if perf_plot:
                    created_files["static_plots"].append(perf_plot)

            # 2. Training progress plots
            if training_history:
                progress_plot = self._create_training_progress(training_history, suite_dir, experiment_name)
                if progress_plot:
                    created_files["static_plots"].append(progress_plot)

                # Training progress animation
                if include_animations:
                    progress_anim = self._create_training_animation(training_history, suite_dir, experiment_name)
                    if progress_anim:
                        created_files["animations"].append(progress_anim)

            # 3. Model comparison matrix
            if comparison_data:
                comparison_plot = self._create_model_comparison(comparison_data, suite_dir, experiment_name)
                if comparison_plot:
                    created_files["static_plots"].append(comparison_plot)

            # 4. Summary dashboard
            summary_plot = self._create_summary_dashboard(experiment_data, suite_dir, experiment_name)
            if summary_plot:
                created_files["static_plots"].append(summary_plot)

            # 5. Performance evolution animation
            if training_history and include_animations:
                evolution_anim = self._create_performance_evolution(training_history, suite_dir, experiment_name)
                if evolution_anim:
                    created_files["animations"].append(evolution_anim)

            # 6. Export visualization data
            data_export = self._export_visualization_data(experiment_data, suite_dir, experiment_name)
            if data_export:
                created_files["data_exports"].append(data_export)

            # 7. Generate HTML report
            html_report = self._generate_html_report(experiment_data, created_files, suite_dir, experiment_name)
            if html_report:
                created_files["reports"].append(html_report)

            self.logger.info(f"Created visualization suite for experiment: {experiment_name}")
            return created_files

        except Exception as e:
            self.logger.error(f"Error creating visualization suite: {e}")
            raise VisualizationError(f"Failed to create visualization suite: {e}")

    def _create_performance_comparison(self, run_results: List[Dict], suite_dir: Path, experiment_name: str) -> Optional[str]:
        """Create performance comparison plot."""
        try:
            # Convert run results to format expected by visualizer
            plot_data = []
            for result in run_results:
                if hasattr(result, 'average_reward'):  # RunResult object
                    plot_data.append({
                        'config_name': result.run_id,
                        'average_reward': result.average_reward,
                        'average_reward_std': 0.0  # Would need multiple runs per config for std
                    })
                else:  # Dictionary
                    plot_data.append(result)

            filename = f"{experiment_name}_performance_comparison.png"
            return self.visualizer.plot_performance_comparison(
                plot_data,
                title=f"{experiment_name} - Performance Comparison",
                filename=filename
            )
        except Exception as e:
            self.logger.error(f"Error creating performance comparison: {e}")
            return None

    def _create_training_progress(self, training_history: List[Dict], suite_dir: Path, experiment_name: str) -> Optional[str]:
        """Create training progress plot."""
        try:
            filename = f"{experiment_name}_training_progress.png"
            return self.visualizer.plot_training_progress(
                training_history,
                title=f"{experiment_name} - Training Progress",
                filename=filename
            )
        except Exception as e:
            self.logger.error(f"Error creating training progress plot: {e}")
            return None

    def _create_training_animation(self, training_history: List[Dict], suite_dir: Path, experiment_name: str) -> Optional[str]:
        """Create training progress animation."""
        try:
            filename = f"{experiment_name}_training_progress.mp4"
            return self.animator.create_training_progress_animation(
                training_history,
                title=f"{experiment_name} - Training Progress",
                filename=filename
            )
        except Exception as e:
            self.logger.error(f"Error creating training animation: {e}")
            return None

    def _create_model_comparison(self, comparison_data: Dict, suite_dir: Path, experiment_name: str) -> Optional[str]:
        """Create model comparison matrix."""
        try:
            filename = f"{experiment_name}_model_comparison.png"
            return self.visualizer.plot_model_comparison_matrix(
                comparison_data,
                title=f"{experiment_name} - Model Comparison",
                filename=filename
            )
        except Exception as e:
            self.logger.error(f"Error creating model comparison: {e}")
            return None

    def _create_summary_dashboard(self, experiment_data: Dict, suite_dir: Path, experiment_name: str) -> Optional[str]:
        """Create summary dashboard."""
        try:
            filename = f"{experiment_name}_summary_dashboard.png"
            return self.visualizer.create_summary_dashboard(
                experiment_data,
                title=f"{experiment_name} - Summary Dashboard",
                filename=filename
            )
        except Exception as e:
            self.logger.error(f"Error creating summary dashboard: {e}")
            return None

    def _create_performance_evolution(self, training_history: List[Dict], suite_dir: Path, experiment_name: str) -> Optional[str]:
        """Create performance evolution animation."""
        try:
            # Convert training history to performance data format
            performance_data = [
                {'performance': entry.get('reward', 0), 'step': entry.get('step', 0)}
                for entry in training_history
            ]

            filename = f"{experiment_name}_performance_evolution.mp4"
            return self.animator.create_performance_evolution_animation(
                performance_data,
                title=f"{experiment_name} - Performance Evolution",
                filename=filename
            )
        except Exception as e:
            self.logger.error(f"Error creating performance evolution: {e}")
            return None

    def _export_visualization_data(self, experiment_data: Dict, suite_dir: Path, experiment_name: str) -> Optional[str]:
        """Export visualization data."""
        try:
            filename = f"{experiment_name}_visualization_data"
            return self.visualizer.export_visualization_data(
                experiment_data, filename, format="json"
            )
        except Exception as e:
            self.logger.error(f"Error exporting visualization data: {e}")
            return None

    def _generate_html_report(self, experiment_data: Dict, created_files: Dict, suite_dir: Path, experiment_name: str) -> Optional[str]:
        """Generate HTML report."""
        try:
            html_content = ".1f"".3f"".1f"".3f"f"""
<!DOCTYPE html>
<html>
<head>
    <title>{experiment_name} - Visualization Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .section {{ margin-bottom: 30px; }}
        .plot-container {{ margin: 20px 0; }}
        img, video {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .summary-box {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>{experiment_name} - Visualization Report</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="section summary-box">
        <h2>Summary</h2>
        <p>Total runs: {experiment_data.get('total_runs', 'N/A')}</p>
        <p>Successful runs: {experiment_data.get('successful_runs', 'N/A')}</p>
        <p>Duration: {experiment_data.get('duration_seconds', 0):.1f}s</p>
    </div>

    <div class="section">
        <h2>Static Plots</h2>
        {"".join(f'<div class="plot-container"><h3>{Path(plot).stem}</h3><img src="{Path(plot).name}" alt="{Path(plot).stem}"></div>' for plot in created_files["static_plots"])}
    </div>

    <div class="section">
        <h2>Animations</h2>
        {"".join(f'<div class="plot-container"><h3>{Path(anim).stem}</h3><video controls><source src="{Path(anim).name}" type="video/mp4">Your browser does not support the video tag.</video></div>' for anim in created_files["animations"])}
    </div>

    <div class="section">
        <h2>Performance Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
"""

            # Add performance metrics
            performance = experiment_data.get('average_performance', {})
            for key, value in performance.items():
                html_content += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value:.3f}</td></tr>"

            html_content += """
        </table>
    </div>
</body>
</html>"""

            html_file = suite_dir / f"{experiment_name}_report.html"
            with open(html_file, 'w') as f:
                f.write(html_content)

            return str(html_file)

        except Exception as e:
            self.logger.error(f"Error generating HTML report: {e}")
            return None

    def batch_create_visualizations(
        self,
        experiments_data: List[Dict[str, Any]],
        experiment_names: List[str],
        include_animations: bool = True,
        theme: str = "default"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Create visualizations for multiple experiments in batch.

        Args:
            experiments_data: List of experiment data dictionaries
            experiment_names: Names for each experiment
            include_animations: Whether to include animations
            theme: Visualization theme

        Returns:
            Dictionary mapping experiment names to their visualization results
        """
        results = {}

        for exp_data, exp_name in zip(experiments_data, experiment_names):
            try:
                self.logger.info(f"Creating visualizations for experiment: {exp_name}")
                exp_results = self.create_experiment_visualization_suite(
                    exp_data, exp_name, include_animations, theme
                )
                results[exp_name] = exp_results

            except Exception as e:
                self.logger.error(f"Failed to create visualizations for {exp_name}: {e}")
                results[exp_name] = {"error": str(e)}

        return results

    def get_visualization_statistics(self) -> Dict[str, Any]:
        """Get statistics about created visualizations."""
        stats = {
            "total_files": 0,
            "static_plots": 0,
            "animations": 0,
            "reports": 0,
            "total_size_mb": 0,
            "by_experiment": {}
        }

        try:
            # Count files in output directory
            for exp_dir in self.output_dir.iterdir():
                if exp_dir.is_dir() and exp_dir.name.endswith('_suite'):
                    exp_stats = {
                        "static_plots": 0,
                        "animations": 0,
                        "reports": 0,
                        "total_size_mb": 0
                    }

                    for file_path in exp_dir.rglob('*'):
                        if file_path.is_file():
                            exp_stats["total_size_mb"] += file_path.stat().st_size / (1024 * 1024)

                            if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.svg']:
                                exp_stats["static_plots"] += 1
                            elif file_path.suffix.lower() in ['.mp4', '.avi', '.gif']:
                                exp_stats["animations"] += 1
                            elif file_path.suffix.lower() in ['.html', '.pdf']:
                                exp_stats["reports"] += 1

                    stats["by_experiment"][exp_dir.name.replace('_suite', '')] = exp_stats
                    stats["static_plots"] += exp_stats["static_plots"]
                    stats["animations"] += exp_stats["animations"]
                    stats["reports"] += exp_stats["reports"]
                    stats["total_size_mb"] += exp_stats["total_size_mb"]

            stats["total_files"] = stats["static_plots"] + stats["animations"] + stats["reports"]

        except Exception as e:
            self.logger.error(f"Error calculating visualization statistics: {e}")

        return stats

    def cleanup_old_visualizations(self, days_old: int = 30) -> int:
        """
        Clean up old visualization files.

        Args:
            days_old: Remove files older than this many days

        Returns:
            Number of files removed
        """
        import time

        current_time = time.time()
        cutoff_time = current_time - (days_old * 24 * 60 * 60)

        removed_count = 0

        try:
            for file_path in self.output_dir.rglob('*'):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    removed_count += 1

            # Remove empty directories
            for dir_path in sorted(self.output_dir.rglob('*'), reverse=True):
                if dir_path.is_dir() and not any(dir_path.iterdir()):
                    dir_path.rmdir()

            self.logger.info(f"Cleaned up {removed_count} old visualization files")
            return removed_count

        except Exception as e:
            self.logger.error(f"Error cleaning up visualizations: {e}")
            return 0
