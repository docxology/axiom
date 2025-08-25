"""
Animator for AXIOM Extensions

Provides animation tools for visualizing AXIOM model training progress,
performance trends, and experimental results over time.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class AnimationError(Exception):
    """Exception raised for animation-related errors."""
    pass


class Animator:
    """
    Creates animations for AXIOM experimental results and training progress.

    This class provides methods to:
    - Generate training progress animations
    - Create performance evolution videos
    - Animate model behavior over time
    - Produce comparative animations
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None, fps: int = 30):
        """
        Initialize animator.

        Args:
            output_dir: Directory to save animations (if None, uses output config)
            fps: Frames per second for animations
        """
        # Import output configuration
        try:
            from ..output_config import output_config
            if output_dir is None:
                # Use centralized output configuration
                self.output_dir = output_config.get_path_for("visualizations", "animations")
            else:
                self.output_dir = Path(output_dir)
        except ImportError:
            # Fallback to default behavior if output_config is not available
            self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "animations"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.logger = logging.getLogger(__name__)

        # Set up animation backend
        self._setup_animation_backend()

    def _setup_animation_backend(self):
        """Set up animation backend (matplotlib or alternative)."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation

            self.matplotlib_available = True
            self.plt = plt
            self.animation = animation
        except ImportError:
            self.logger.warning("Matplotlib not available, animation features disabled")
            self.matplotlib_available = False

    def create_training_progress_animation(
        self,
        training_history: List[Dict[str, Any]],
        metrics: List[str] = None,
        title: str = "Training Progress Animation",
        filename: str = None,
        duration: float = 10.0,
        figsize: Tuple[float, float] = (12, 8)
    ) -> Optional[str]:
        """
        Create an animation of training progress over time.

        Args:
            training_history: List of training step data
            metrics: List of metrics to animate
            title: Animation title
            filename: Output filename
            duration: Animation duration in seconds
            figsize: Figure size

        Returns:
            Path to saved animation file, or None if animation unavailable
        """
        if not self.matplotlib_available:
            self.logger.warning("Matplotlib not available, cannot create animations")
            return None

        if not training_history:
            raise AnimationError("No training history provided")

        if metrics is None:
            metrics = ["reward", "loss"]

        if filename is None:
            filename = "training_progress_animation.mp4"

        try:
            fig, axes = self.plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)

            if len(metrics) == 1:
                axes = [axes]

            # Extract data
            steps = [entry.get('step', i) for i, entry in enumerate(training_history)]
            metric_data = {}

            for metric in metrics:
                data = []
                for entry in training_history:
                    data.append(entry.get(metric, 0))
                metric_data[metric] = data

            # Calculate animation parameters
            total_frames = int(duration * self.fps)
            step_increment = max(1, len(steps) // total_frames)

            def animate(frame):
                current_step = min(frame * step_increment, len(steps) - 1)

                for ax, metric in zip(axes, metrics):
                    ax.clear()
                    data = metric_data[metric][:current_step + 1]
                    current_steps = steps[:current_step + 1]

                    if current_steps and data:
                        ax.plot(current_steps, data, 'b-', marker='o', markersize=3, alpha=0.7)
                        ax.set_ylabel(metric.replace('_', ' ').title())
                        ax.grid(True, alpha=0.3)

                        # Add current value annotation
                        if data:
                            ax.text(0.02, 0.98, f'{metric.title()}: {data[-1]:.3f}',
                                  transform=ax.transAxes, verticalalignment='top',
                                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                axes[-1].set_xlabel('Training Step')
                fig.suptitle(f"{title} (Step {current_step})")

                self.plt.tight_layout()

            # Create animation
            anim = self.animation.FuncAnimation(
                fig, animate, frames=total_frames, interval=1000/self.fps, repeat=False
            )

            # Save animation
            filepath = self.output_dir / filename

            # Try different writers
            writers = ['ffmpeg', 'avconv', 'mencoder']
            saved = False

            for writer in writers:
                try:
                    Writer = self.animation.writers[writer]
                    writer_obj = Writer(fps=self.fps, metadata=dict(artist='AXIOM'))
                    anim.save(filepath, writer=writer_obj)
                    saved = True
                    self.logger.info(f"Saved animation using {writer} writer")
                    break
                except (KeyError, RuntimeError) as e:
                    self.logger.debug(f"Writer {writer} not available: {e}")
                    continue

            if not saved:
                # Fallback: save as GIF if pillow available
                try:
                    import PIL
                    anim.save(filepath.with_suffix('.gif'), writer='pillow', fps=self.fps)
                    filepath = filepath.with_suffix('.gif')
                    saved = True
                    self.logger.info("Saved animation as GIF")
                except ImportError:
                    self.logger.warning("No animation writers available, cannot save animation")

            self.plt.close(fig)

            if saved:
                self.logger.info(f"Saved training progress animation to: {filepath}")
                return str(filepath)
            else:
                return None

        except Exception as e:
            self.logger.error(f"Error creating training progress animation: {e}")
            if 'fig' in locals():
                self.plt.close('all')
            raise AnimationError(f"Failed to create animation: {e}")

    def create_performance_evolution_animation(
        self,
        performance_data: List[Dict[str, Any]],
        title: str = "Performance Evolution",
        filename: str = None,
        duration: float = 8.0,
        figsize: Tuple[float, float] = (10, 6)
    ) -> Optional[str]:
        """
        Create an animation showing performance evolution over time.

        Args:
            performance_data: List of performance snapshots
            title: Animation title
            filename: Output filename
            duration: Animation duration in seconds
            figsize: Figure size

        Returns:
            Path to saved animation file, or None if animation unavailable
        """
        if not self.matplotlib_available:
            return None

        if not performance_data:
            raise AnimationError("No performance data provided")

        if filename is None:
            filename = "performance_evolution.mp4"

        try:
            fig, ax = self.plt.subplots(figsize=figsize)

            # Extract data
            time_points = list(range(len(performance_data)))
            performances = [data.get('performance', 0) for data in performance_data]
            best_performances = []

            current_best = float('-inf')
            for perf in performances:
                current_best = max(current_best, perf)
                best_performances.append(current_best)

            # Calculate animation parameters
            total_frames = int(duration * self.fps)
            data_increment = max(1, len(time_points) // total_frames)

            def animate(frame):
                ax.clear()

                current_idx = min(frame * data_increment, len(time_points) - 1)
                current_time = time_points[:current_idx + 1]
                current_perf = performances[:current_idx + 1]
                current_best = best_performances[:current_idx + 1]

                if current_time and current_perf:
                    # Plot current performance
                    ax.plot(current_time, current_perf, 'b-', alpha=0.7, label='Current')
                    ax.plot(current_time, current_best, 'r--', alpha=0.8, label='Best')

                    # Add markers
                    ax.scatter(current_time[-1], current_perf[-1], c='blue', s=50, zorder=5)
                    ax.scatter(current_time[-1], current_best[-1], c='red', s=50, zorder=5)

                    ax.set_xlabel('Time Step')
                    ax.set_ylabel('Performance')
                    ax.set_title(f"{title} (Time: {current_time[-1]})")
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                    # Add statistics text
                    stats_text = f"""Current: {current_perf[-1]:.3f}
Best: {current_best[-1]:.3f}
Improvement: {current_best[-1] - current_perf[0]:.3f}"""

                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                           verticalalignment='top', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Create animation
            anim = self.animation.FuncAnimation(
                fig, animate, frames=total_frames, interval=1000/self.fps, repeat=False
            )

            # Save animation
            filepath = self.output_dir / filename
            saved = self._save_animation(anim, filepath)

            self.plt.close(fig)

            if saved:
                self.logger.info(f"Saved performance evolution animation to: {filepath}")
                return str(filepath)
            else:
                return None

        except Exception as e:
            self.logger.error(f"Error creating performance evolution animation: {e}")
            if 'fig' in locals():
                self.plt.close('all')
            raise AnimationError(f"Failed to create animation: {e}")

    def create_comparison_animation(
        self,
        comparison_data: Dict[str, List[Dict[str, Any]]],
        title: str = "Model Comparison Animation",
        filename: str = None,
        duration: float = 12.0,
        figsize: Tuple[float, float] = (12, 8)
    ) -> Optional[str]:
        """
        Create an animation comparing multiple models over time.

        Args:
            comparison_data: Dictionary mapping model names to their performance data
            title: Animation title
            filename: Output filename
            duration: Animation duration in seconds
            figsize: Figure size

        Returns:
            Path to saved animation file, or None if animation unavailable
        """
        if not self.matplotlib_available:
            return None

        if not comparison_data:
            raise AnimationError("No comparison data provided")

        if filename is None:
            filename = "model_comparison_animation.mp4"

        try:
            fig, ax = self.plt.subplots(figsize=figsize)

            model_names = list(comparison_data.keys())
            max_length = max(len(data) for data in comparison_data.values())

            # Prepare data
            model_data = {}
            for name, data in comparison_data.items():
                performances = [d.get('performance', 0) for d in data]
                # Pad shorter sequences
                if len(performances) < max_length:
                    performances.extend([performances[-1]] * (max_length - len(performances)))
                model_data[name] = performances

            time_points = list(range(max_length))

            # Calculate animation parameters
            total_frames = int(duration * self.fps)
            data_increment = max(1, max_length // total_frames)

            def animate(frame):
                ax.clear()

                current_idx = min(frame * data_increment, max_length - 1)

                colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
                best_performance = float('-inf')
                best_model = None

                for i, (name, performances) in enumerate(model_data.items()):
                    color = colors[i % len(colors)]
                    current_data = performances[:current_idx + 1]
                    current_time = time_points[:current_idx + 1]

                    ax.plot(current_time, current_data, color=color, label=name,
                           marker='o', markersize=3, alpha=0.8)

                    # Track best performer
                    if current_data and current_data[-1] > best_performance:
                        best_performance = current_data[-1]
                        best_model = name

                ax.set_xlabel('Time Step')
                ax.set_ylabel('Performance')
                ax.set_title(f"{title} (Time: {current_idx})")
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Add current rankings
                if current_idx >= 0:
                    current_performances = [(name, data[current_idx])
                                          for name, data in model_data.items()]
                    current_performances.sort(key=lambda x: x[1], reverse=True)

                    ranking_text = "Current Ranking:\n"
                    for rank, (name, perf) in enumerate(current_performances[:3], 1):
                        marker = "👑" if name == best_model else f"{rank}."
                        ranking_text += ".3f"

                    ax.text(0.02, 0.98, ranking_text, transform=ax.transAxes,
                           verticalalignment='top', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

            # Create animation
            anim = self.animation.FuncAnimation(
                fig, animate, frames=total_frames, interval=1000/self.fps, repeat=False
            )

            # Save animation
            filepath = self.output_dir / filename
            saved = self._save_animation(anim, filepath)

            self.plt.close(fig)

            if saved:
                self.logger.info(f"Saved comparison animation to: {filepath}")
                return str(filepath)
            else:
                return None

        except Exception as e:
            self.logger.error(f"Error creating comparison animation: {e}")
            if 'fig' in locals():
                self.plt.close('all')
            raise AnimationError(f"Failed to create animation: {e}")

    def _save_animation(self, anim, filepath: Path) -> bool:
        """Save animation using available writers."""
        # Try different writers
        writers = ['ffmpeg', 'avconv', 'mencoder']
        saved = False

        for writer in writers:
            try:
                Writer = self.animation.writers[writer]
                writer_obj = Writer(fps=self.fps, metadata=dict(artist='AXIOM'))
                anim.save(filepath, writer=writer_obj)
                saved = True
                self.logger.info(f"Saved animation using {writer} writer")
                break
            except (KeyError, RuntimeError) as e:
                self.logger.debug(f"Writer {writer} not available: {e}")
                continue

        if not saved:
            # Fallback: save as GIF if pillow available
            try:
                import PIL
                gif_path = filepath.with_suffix('.gif')
                anim.save(gif_path, writer='pillow', fps=self.fps)
                # Update filepath to point to GIF
                import shutil
                shutil.move(gif_path, filepath)
                saved = True
                self.logger.info("Saved animation as GIF")
            except (ImportError, Exception) as e:
                self.logger.warning(f"No animation writers available: {e}")

        return saved

    def create_custom_animation(
        self,
        update_function: Callable[[int], Any],
        total_frames: int,
        title: str = "Custom Animation",
        filename: str = None,
        figsize: Tuple[float, float] = (10, 6)
    ) -> Optional[str]:
        """
        Create a custom animation using a user-provided update function.

        Args:
            update_function: Function called for each frame, takes frame number
            total_frames: Total number of frames
            title: Animation title
            filename: Output filename
            figsize: Figure size

        Returns:
            Path to saved animation file, or None if animation unavailable
        """
        if not self.matplotlib_available:
            return None

        if filename is None:
            filename = "custom_animation.mp4"

        try:
            fig, ax = self.plt.subplots(figsize=figsize)

            def animate(frame):
                ax.clear()
                result = update_function(frame)
                ax.set_title(f"{title} (Frame {frame})")
                return result

            # Create animation
            anim = self.animation.FuncAnimation(
                fig, animate, frames=total_frames, interval=1000/self.fps, repeat=False
            )

            # Save animation
            filepath = self.output_dir / filename
            saved = self._save_animation(anim, filepath)

            self.plt.close(fig)

            if saved:
                self.logger.info(f"Saved custom animation to: {filepath}")
                return str(filepath)
            else:
                return None

        except Exception as e:
            self.logger.error(f"Error creating custom animation: {e}")
            if 'fig' in locals():
                self.plt.close('all')
            raise AnimationError(f"Failed to create animation: {e}")

    def get_supported_formats(self) -> List[str]:
        """Get list of supported animation formats."""
        if not self.matplotlib_available:
            return []

        formats = []
        try:
            for writer in self.animation.writers:
                try:
                    self.animation.writers[writer]
                    if writer == 'ffmpeg':
                        formats.extend(['mp4', 'avi', 'mov'])
                    elif writer == 'pillow':
                        formats.append('gif')
                except KeyError:
                    continue
        except Exception:
            pass

        return formats if formats else ['gif']  # Fallback

    def set_fps(self, fps: int) -> None:
        """Set animation frames per second."""
        if fps <= 0:
            raise ValueError("FPS must be positive")
        self.fps = fps
        self.logger.info(f"Set animation FPS to: {fps}")
