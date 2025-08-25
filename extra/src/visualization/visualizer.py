"""
Visualizer for AXIOM Extensions

Provides static visualization tools for AXIOM model performance,
experimental results, and comparative analysis.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class VisualizationError(Exception):
    """Exception raised for visualization-related errors."""
    pass


class Visualizer:
    """
    Creates static visualizations for AXIOM experimental results.

    This class provides methods to:
    - Generate performance comparison plots
    - Create model architecture diagrams
    - Visualize training progress
    - Produce comparative analysis charts
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None, style: str = "default"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save visualizations (if None, uses output config)
            style: Plot style theme ("default", "dark", "minimal")
        """
        # Import output configuration
        try:
            from ..output_config import output_config
            if output_dir is None:
                # Use centralized output configuration
                self.output_dir = output_config.get_path_for("visualizations", "static")
            else:
                self.output_dir = Path(output_dir)
        except ImportError:
            # Fallback to default behavior if output_config is not available
            self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "visualizations"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.style = style
        self.logger = logging.getLogger(__name__)

        # Set up plotting backend
        self._setup_plotting_backend()

    def _setup_plotting_backend(self):
        """Set up matplotlib backend for headless operation."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt

            # Set style
            if self.style == "dark":
                plt.style.use('dark_background')
            elif self.style == "minimal":
                plt.style.use('seaborn-v0_8-white') if hasattr(plt.style, 'available') and 'seaborn-v0_8-white' in plt.style.available else plt.style.use('default')

            self.matplotlib_available = True
            self.plt = plt
        except ImportError:
            self.logger.warning("Matplotlib not available, visualization features disabled")
            self.matplotlib_available = False

    def plot_performance_comparison(
        self,
        results: List[Dict[str, Any]],
        metric: str = "average_reward",
        title: str = "Performance Comparison",
        filename: Optional[str] = None,
        figsize: Tuple[float, float] = (10, 6)
    ) -> Optional[str]:
        """
        Create a performance comparison plot.

        Args:
            results: List of result dictionaries with performance data
            metric: Metric to plot (default: "average_reward")
            title: Plot title
            filename: Optional filename (auto-generated if None)
            figsize: Figure size as (width, height)

        Returns:
            Path to saved plot file, or None if plotting unavailable
        """
        if not self.matplotlib_available:
            self.logger.warning("Matplotlib not available, cannot create plots")
            return None

        if not results:
            raise VisualizationError("No results provided for plotting")

        try:
            fig, ax = self.plt.subplots(figsize=figsize)

            # Extract data
            labels = []
            values = []
            errors = []

            for result in results:
                config_name = result.get('config_name', f'Config {len(labels)}')
                labels.append(config_name)

                if metric in result:
                    values.append(result[metric])
                    # Add error bars if available
                    error_metric = f"{metric}_std"
                    if error_metric in result:
                        errors.append(result[error_metric])
                    else:
                        errors.append(0)
                else:
                    self.logger.warning(f"Metric '{metric}' not found in result: {result}")
                    values.append(0)
                    errors.append(0)

            # Create bar plot
            x_pos = np.arange(len(labels))
            bars = ax.bar(x_pos, values, yerr=errors, capsize=5, alpha=0.7)

            # Customize plot
            ax.set_xlabel('Configuration')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(title)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=45, ha='right')

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(errors) * 0.1,
                       f'{value:.2f}', ha='center', va='bottom')

            self.plt.tight_layout()

            # Save plot
            if filename is None:
                filename = f"performance_comparison_{metric}.png"

            filepath = self.output_dir / filename
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            self.plt.close(fig)

            self.logger.info(f"Saved performance comparison plot to: {filepath}")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"Error creating performance comparison plot: {e}")
            if 'fig' in locals():
                self.plt.close('all')
            raise VisualizationError(f"Failed to create plot: {e}")

    def plot_training_progress(
        self,
        training_history: List[Dict[str, Any]],
        metrics: Optional[List[str]] = None,
        title: str = "Training Progress",
        filename: Optional[str] = None,
        figsize: Tuple[float, float] = (12, 8)
    ) -> Optional[str]:
        """
        Create a training progress plot with multiple metrics.

        Args:
            training_history: List of training step data
            metrics: List of metrics to plot (default: ["reward", "loss"])
            title: Plot title
            filename: Optional filename
            figsize: Figure size

        Returns:
            Path to saved plot file, or None if plotting unavailable
        """
        if not self.matplotlib_available:
            return None

        if not training_history:
            raise VisualizationError("No training history provided")

        if metrics is None:
            metrics = ["reward", "loss"]

        try:
            fig, axes = self.plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)

            if len(metrics) == 1:
                axes = [axes]

            steps = [entry.get('step', i) for i, entry in enumerate(training_history)]

            for ax, metric in zip(axes, metrics):
                values = []
                for entry in training_history:
                    if metric in entry:
                        values.append(entry[metric])
                    else:
                        values.append(0)  # Default value

                ax.plot(steps, values, marker='o', markersize=2, alpha=0.7)
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.grid(True, alpha=0.3)

                # Add trend line if enough data
                if len(values) > 5:
                    try:
                        z = np.polyfit(steps, values, 1)
                        p = np.poly1d(z)
                        ax.plot(steps, p(steps), "r--", alpha=0.8, label="Trend")
                        ax.legend()
                    except Exception:
                        pass  # Skip trend line if fitting fails

            axes[-1].set_xlabel('Training Step')
            fig.suptitle(title)

            self.plt.tight_layout()

            # Save plot
            if filename is None:
                filename = "training_progress.png"

            filepath = self.output_dir / filename
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            self.plt.close(fig)

            self.logger.info(f"Saved training progress plot to: {filepath}")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"Error creating training progress plot: {e}")
            if 'fig' in locals():
                self.plt.close('all')
            raise VisualizationError(f"Failed to create plot: {e}")

    def plot_model_comparison_matrix(
        self,
        comparison_data: Dict[str, Dict[str, Any]],
        metrics: List[str] = None,
        title: str = "Model Comparison Matrix",
        filename: Optional[str] = None,
        figsize: Tuple[float, float] = (12, 8)
    ) -> Optional[str]:
        """
        Create a comparison matrix showing multiple models across multiple metrics.

        Args:
            comparison_data: Dictionary mapping model names to their metrics
            metrics: List of metrics to include
            title: Plot title
            filename: Optional filename
            figsize: Figure size

        Returns:
            Path to saved plot file, or None if plotting unavailable
        """
        if not self.matplotlib_available:
            return None

        if not comparison_data:
            raise VisualizationError("No comparison data provided")

        if metrics is None:
            # Extract common metrics from all models
            all_metrics = set()
            for model_data in comparison_data.values():
                all_metrics.update(model_data.keys())
            metrics = sorted(list(all_metrics))

        try:
            fig, ax = self.plt.subplots(figsize=figsize)

            model_names = list(comparison_data.keys())
            n_models = len(model_names)
            n_metrics = len(metrics)

            # Create data matrix
            data_matrix = np.zeros((n_models, n_metrics))

            for i, model in enumerate(model_names):
                for j, metric in enumerate(metrics):
                    data_matrix[i, j] = comparison_data[model].get(metric, 0)

            # Create heatmap
            im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto')

            # Set labels
            ax.set_xticks(np.arange(n_metrics))
            ax.set_yticks(np.arange(n_models))
            ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
            ax.set_yticklabels(model_names)

            # Rotate x-axis labels
            self.plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel("Value", rotation=-90, va="bottom")

            # Add value annotations
            for i in range(n_models):
                for j in range(n_metrics):
                    text = ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black")

            ax.set_title(title)
            fig.tight_layout()

            # Save plot
            if filename is None:
                filename = "model_comparison_matrix.png"

            filepath = self.output_dir / filename
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            self.plt.close(fig)

            self.logger.info(f"Saved model comparison matrix to: {filepath}")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"Error creating model comparison matrix: {e}")
            if 'fig' in locals():
                self.plt.close('all')
            raise VisualizationError(f"Failed to create plot: {e}")

    def create_summary_dashboard(
        self,
        summary_data: Dict[str, Any],
        title: str = "Experiment Summary Dashboard",
        filename: Optional[str] = None,
        figsize: Tuple[float, float] = (15, 10)
    ) -> Optional[str]:
        """
        Create a comprehensive summary dashboard.

        Args:
            summary_data: Dictionary with summary statistics
            title: Dashboard title
            filename: Optional filename
            figsize: Figure size

        Returns:
            Path to saved dashboard file, or None if plotting unavailable
        """
        if not self.matplotlib_available:
            return None

        if not summary_data:
            raise VisualizationError("No summary data provided")

        try:
            fig, axes = self.plt.subplots(2, 2, figsize=figsize)
            fig.suptitle(title, fontsize=16)

            # Performance overview
            if 'performance' in summary_data:
                perf_data = summary_data['performance']
                axes[0, 0].bar(['Mean', 'Best', 'Worst'],
                             [perf_data.get('mean', 0), perf_data.get('best', 0), perf_data.get('worst', 0)])
                axes[0, 0].set_title('Performance Overview')
                axes[0, 0].set_ylabel('Reward')

            # Stability metrics
            if 'stability' in summary_data:
                stability_data = summary_data['stability']
                metrics = ['Consistency', 'Variance', 'Range']
                values = [
                    stability_data.get('consistency', 0),
                    stability_data.get('variance', 0),
                    stability_data.get('range', 0)
                ]
                axes[0, 1].bar(metrics, values)
                axes[0, 1].set_title('Stability Metrics')

            # Convergence analysis
            if 'convergence' in summary_data:
                conv_data = summary_data['convergence']
                axes[1, 0].pie([conv_data.get('converged', 0), conv_data.get('total', 1) - conv_data.get('converged', 0)],
                             labels=['Converged', 'Not Converged'], autopct='%1.1f%%')
                axes[1, 0].set_title('Convergence Analysis')

            # Resource usage
            if 'resources' in summary_data:
                resource_data = summary_data['resources']
                axes[1, 1].bar(['CPU', 'Memory', 'GPU'],
                             [resource_data.get('cpu', 0), resource_data.get('memory', 0), resource_data.get('gpu', 0)])
                axes[1, 1].set_title('Resource Usage')
                axes[1, 1].set_ylabel('Usage %')

            self.plt.tight_layout()

            # Save dashboard
            if filename is None:
                filename = "summary_dashboard.png"

            filepath = self.output_dir / filename
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            self.plt.close(fig)

            self.logger.info(f"Saved summary dashboard to: {filepath}")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"Error creating summary dashboard: {e}")
            if 'fig' in locals():
                self.plt.close('all')
            raise VisualizationError(f"Failed to create dashboard: {e}")

    def export_visualization_data(
        self,
        data: Dict[str, Any],
        filename: str,
        format: str = "json"
    ) -> str:
        """
        Export visualization data for external processing.

        Args:
            data: Data to export
            filename: Output filename (without extension)
            format: Export format ("json", "csv")

        Returns:
            Path to exported file
        """
        try:
            if format == "json":
                import json
                filepath = self.output_dir / f"{filename}.json"
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            elif format == "csv":
                import csv
                filepath = self.output_dir / f"{filename}.csv"
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # Simple CSV export for basic data structures
                    if isinstance(data, dict):
                        writer.writerow(['Key', 'Value'])
                        for key, value in data.items():
                            writer.writerow([key, value])
                    elif isinstance(data, list) and data and isinstance(data[0], dict):
                        if data:
                            writer.writerow(data[0].keys())
                            for row in data:
                                writer.writerow(row.values())
            else:
                raise ValueError(f"Unsupported export format: {format}")

            self.logger.info(f"Exported visualization data to: {filepath}")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"Error exporting visualization data: {e}")
            raise VisualizationError(f"Failed to export data: {e}")

    def get_available_styles(self) -> List[str]:
        """Get list of available visualization styles."""
        if not self.matplotlib_available:
            return []

        try:
            return self.plt.style.available
        except Exception:
            return ["default"]

    def set_style(self, style: str) -> None:
        """
        Set the visualization style.

        Args:
            style: Style name
        """
        if not self.matplotlib_available:
            self.logger.warning("Matplotlib not available, cannot set style")
            return

        try:
            self.plt.style.use(style)
            self.style = style
            self.logger.info(f"Set visualization style to: {style}")
        except Exception as e:
            self.logger.error(f"Error setting style '{style}': {e}")
            self.plt.style.use('default')
            self.style = 'default'
