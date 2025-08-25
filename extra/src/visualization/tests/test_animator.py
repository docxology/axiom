"""
Tests for Animator

Comprehensive tests for the Animator class.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the src directory to the path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visualization.animator import Animator, AnimationError


class TestAnimator(unittest.TestCase):
    """Tests for Animator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.animator = Animator(self.temp_dir, fps=15)  # Lower FPS for testing

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test Animator initialization."""
        animator = Animator("/tmp/test_anim", fps=30)
        self.assertIsNotNone(animator.output_dir)
        self.assertEqual(animator.fps, 30)

    def test_matplotlib_unavailable(self):
        """Test behavior when matplotlib is not available."""
        with patch.dict('sys.modules', {'matplotlib': None, 'matplotlib.pyplot': None}):
            animator = Animator("/tmp/test_anim")
            self.assertFalse(animator.matplotlib_available)

    @patch.object(Animator, 'matplotlib_available', True)
    def test_create_training_progress_animation(self):
        """Test creating training progress animation."""
        training_history = [
            {"step": 0, "reward": 50.0, "loss": 2.0},
            {"step": 10, "reward": 60.0, "loss": 1.8},
            {"step": 20, "reward": 70.0, "loss": 1.5}
        ]

        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.animation.FuncAnimation') as mock_anim:

            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, [mock_ax, mock_ax])

            mock_animation = MagicMock()
            mock_anim.return_value = mock_animation
            mock_animation.save = MagicMock()

            result = self.animator.create_training_progress_animation(
                training_history,
                title="Test Training Animation",
                filename="test_training.mp4",
                duration=2.0  # Short duration for testing
            )

            expected_path = Path(self.temp_dir) / "test_training.mp4"
            self.assertEqual(result, str(expected_path))

    def test_create_training_progress_animation_no_data(self):
        """Test creating animation with no data."""
        with self.assertRaises(AnimationError):
            self.animator.create_training_progress_animation([])

    def test_create_training_progress_animation_no_matplotlib(self):
        """Test creating animation when matplotlib is unavailable."""
        with patch.object(self.animator, 'matplotlib_available', False):
            result = self.animator.create_training_progress_animation(
                [{"step": 0, "reward": 50.0}]
            )
            self.assertIsNone(result)

    @patch.object(Animator, 'matplotlib_available', True)
    def test_create_performance_evolution_animation(self):
        """Test creating performance evolution animation."""
        performance_data = [
            {"performance": 50.0, "step": 0},
            {"performance": 60.0, "step": 100},
            {"performance": 70.0, "step": 200}
        ]

        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.animation.FuncAnimation') as mock_anim:

            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            mock_animation = MagicMock()
            mock_anim.return_value = mock_animation
            mock_animation.save = MagicMock()

            result = self.animator.create_performance_evolution_animation(
                performance_data,
                title="Test Performance Evolution",
                filename="test_evolution.mp4",
                duration=2.0
            )

            expected_path = Path(self.temp_dir) / "test_evolution.mp4"
            self.assertEqual(result, str(expected_path))

    @patch.object(Animator, 'matplotlib_available', True)
    def test_create_comparison_animation(self):
        """Test creating model comparison animation."""
        comparison_data = {
            "model1": [{"performance": 50.0}, {"performance": 60.0}],
            "model2": [{"performance": 55.0}, {"performance": 65.0}]
        }

        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.animation.FuncAnimation') as mock_anim:

            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            mock_animation = MagicMock()
            mock_anim.return_value = mock_animation
            mock_animation.save = MagicMock()

            result = self.animator.create_comparison_animation(
                comparison_data,
                title="Test Comparison Animation",
                filename="test_comparison.mp4",
                duration=2.0
            )

            expected_path = Path(self.temp_dir) / "test_comparison.mp4"
            self.assertEqual(result, str(expected_path))

    def test_create_comparison_animation_no_data(self):
        """Test creating comparison animation with no data."""
        with self.assertRaises(AnimationError):
            self.animator.create_comparison_animation({})

    @patch.object(Animator, 'matplotlib_available', True)
    def test_create_custom_animation(self):
        """Test creating custom animation."""
        def update_func(frame):
            return f"Frame {frame}"

        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.animation.FuncAnimation') as mock_anim:

            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            mock_animation = MagicMock()
            mock_anim.return_value = mock_animation
            mock_animation.save = MagicMock()

            result = self.animator.create_custom_animation(
                update_func,
                total_frames=10,
                title="Test Custom Animation",
                filename="test_custom.mp4"
            )

            expected_path = Path(self.temp_dir) / "test_custom.mp4"
            self.assertEqual(result, str(expected_path))

    def test_get_supported_formats(self):
        """Test getting supported animation formats."""
        with patch.object(self.animator, 'matplotlib_available', True):
            with patch('matplotlib.animation.writers', {'ffmpeg': MagicMock(), 'pillow': MagicMock()}):
                formats = self.animator.get_supported_formats()
                self.assertIn('mp4', formats)
                self.assertIn('gif', formats)

    def test_get_supported_formats_no_matplotlib(self):
        """Test getting supported formats when matplotlib is unavailable."""
        with patch.object(self.animator, 'matplotlib_available', False):
            formats = self.animator.get_supported_formats()
            self.assertEqual(formats, [])

    def test_set_fps(self):
        """Test setting animation FPS."""
        self.animator.set_fps(60)
        self.assertEqual(self.animator.fps, 60)

    def test_set_fps_invalid(self):
        """Test setting invalid FPS."""
        with self.assertRaises(ValueError):
            self.animator.set_fps(0)

        with self.assertRaises(ValueError):
            self.animator.set_fps(-5)

    @patch.object(Animator, 'matplotlib_available', True)
    def test_save_animation_fallback_to_gif(self):
        """Test animation saving with fallback to GIF."""
        mock_anim = MagicMock()

        # Mock all writers as unavailable except pillow
        with patch('matplotlib.animation.writers', {}):
            with patch('matplotlib.animation.writers.__contains__', return_value=False):
                with patch('builtins.__import__') as mock_import:
                    # Mock PIL import
                    mock_pil = MagicMock()
                    mock_import.return_value = mock_pil

                    # Mock shutil.move
                    with patch('shutil.move') as mock_move:
                        result = self.animator._save_animation(mock_anim, Path("/tmp/test.mp4"))
                        self.assertTrue(result)

    @patch.object(Animator, 'matplotlib_available', True)
    def test_save_animation_no_writers(self):
        """Test animation saving when no writers are available."""
        mock_anim = MagicMock()

        with patch('matplotlib.animation.writers', {}):
            with patch('builtins.__import__', side_effect=ImportError("PIL not available")):
                result = self.animator._save_animation(mock_anim, Path("/tmp/test.mp4"))
                self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
