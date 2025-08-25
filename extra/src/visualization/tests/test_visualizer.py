"""
Tests for Visualizer

Comprehensive tests for the Visualizer class.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the src directory to the path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visualization.visualizer import Visualizer, VisualizationError


class TestVisualizer(unittest.TestCase):
    """Tests for Visualizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = Visualizer(self.temp_dir, style="default")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test Visualizer initialization."""
        visualizer = Visualizer("/tmp/test_viz")
        self.assertIsNotNone(visualizer.output_dir)
        self.assertEqual(visualizer.style, "default")

    def test_initialization_with_custom_style(self):
        """Test Visualizer initialization with custom style."""
        visualizer = Visualizer("/tmp/test_viz", style="dark")
        self.assertEqual(visualizer.style, "dark")

    def test_matplotlib_unavailable(self):
        """Test behavior when matplotlib is not available."""
        with patch.dict('sys.modules', {'matplotlib': None, 'matplotlib.pyplot': None}):
            visualizer = Visualizer("/tmp/test_viz")
            self.assertFalse(visualizer.matplotlib_available)

    @patch.object(Visualizer, 'matplotlib_available', True)
    def test_plot_performance_comparison(self):
        """Test performance comparison plotting."""
        test_data = [
            {"config_name": "config1", "average_reward": 100.0, "average_reward_std": 5.0},
            {"config_name": "config2", "average_reward": 120.0, "average_reward_std": 8.0},
            {"config_name": "config3", "average_reward": 90.0, "average_reward_std": 3.0}
        ]

        # Mock matplotlib components
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout') as mock_layout:

            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            # Mock savefig to avoid actual file operations
            mock_fig.savefig = MagicMock()

            result = self.visualizer.plot_performance_comparison(
                test_data,
                title="Test Comparison",
                filename="test_comparison.png"
            )

            # Verify the result path is correct
            expected_path = Path(self.temp_dir) / "test_comparison.png"
            self.assertEqual(result, str(expected_path))

            # Verify matplotlib calls
            mock_subplots.assert_called_once()
            mock_ax.bar.assert_called_once()
            mock_ax.set_xlabel.assert_called_once_with('Configuration')
            mock_ax.set_ylabel.assert_called_once_with('Average Reward')
            mock_ax.set_title.assert_called_once_with('Test Comparison')
            mock_fig.savefig.assert_called_once()

    def test_plot_performance_comparison_no_data(self):
        """Test performance comparison with no data."""
        with self.assertRaises(VisualizationError):
            self.visualizer.plot_performance_comparison([])

    def test_plot_performance_comparison_missing_matplotlib(self):
        """Test performance comparison when matplotlib is unavailable."""
        with patch.object(self.visualizer, 'matplotlib_available', False):
            result = self.visualizer.plot_performance_comparison([{"config_name": "test", "average_reward": 100.0}])
            self.assertIsNone(result)

    @patch.object(Visualizer, 'matplotlib_available', True)
    def test_plot_training_progress(self):
        """Test training progress plotting."""
        training_history = [
            {"step": 0, "reward": 50.0, "loss": 2.0},
            {"step": 100, "reward": 60.0, "loss": 1.8},
            {"step": 200, "reward": 70.0, "loss": 1.5}
        ]

        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, [mock_ax])
            mock_fig.savefig = MagicMock()

            result = self.visualizer.plot_training_progress(
                training_history,
                metrics=["reward", "loss"],
                filename="test_training.png"
            )

            expected_path = Path(self.temp_dir) / "test_training.png"
            self.assertEqual(result, str(expected_path))

    @patch.object(Visualizer, 'matplotlib_available', True)
    def test_plot_model_comparison_matrix(self):
        """Test model comparison matrix plotting."""
        comparison_data = {
            "model1": {"accuracy": 0.85, "loss": 0.3, "speed": 100},
            "model2": {"accuracy": 0.90, "loss": 0.2, "speed": 80},
            "model3": {"accuracy": 0.80, "loss": 0.4, "speed": 120}
        }

        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            mock_fig.savefig = MagicMock()

            result = self.visualizer.plot_model_comparison_matrix(
                comparison_data,
                title="Test Comparison",
                filename="test_matrix.png"
            )

            expected_path = Path(self.temp_dir) / "test_matrix.png"
            self.assertEqual(result, str(expected_path))

    @patch.object(Visualizer, 'matplotlib_available', True)
    def test_create_summary_dashboard(self):
        """Test summary dashboard creation."""
        summary_data = {
            "performance": {"mean": 100.0, "best": 120.0, "worst": 80.0},
            "stability": {"consistency": 0.9, "variance": 0.1, "range": 40.0},
            "convergence": {"converged": 2, "total": 3}
        }

        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_axes = [MagicMock() for _ in range(4)]
            mock_subplots.return_value = (mock_fig, mock_axes)
            mock_fig.savefig = MagicMock()

            result = self.visualizer.create_summary_dashboard(
                summary_data,
                title="Test Dashboard",
                filename="test_dashboard.png"
            )

            expected_path = Path(self.temp_dir) / "test_dashboard.png"
            self.assertEqual(result, str(expected_path))

    def test_export_visualization_data_json(self):
        """Test exporting visualization data as JSON."""
        test_data = {"experiment": "test", "metrics": [1, 2, 3, 4, 5]}

        result = self.visualizer.export_visualization_data(
            test_data, "test_export", format="json"
        )

        expected_path = Path(self.temp_dir) / "test_export.json"
        self.assertEqual(result, str(expected_path))

        # Verify file was created and contains correct data
        self.assertTrue(expected_path.exists())
        with open(expected_path, 'r') as f:
            loaded_data = json.load(f)
        self.assertEqual(loaded_data, test_data)

    def test_export_visualization_data_csv(self):
        """Test exporting visualization data as CSV."""
        test_data = [
            {"name": "config1", "value": 100},
            {"name": "config2", "value": 120}
        ]

        result = self.visualizer.export_visualization_data(
            test_data, "test_export", format="csv"
        )

        expected_path = Path(self.temp_dir) / "test_export.csv"
        self.assertEqual(result, str(expected_path))

        # Verify file was created
        self.assertTrue(expected_path.exists())

    def test_export_visualization_data_invalid_format(self):
        """Test exporting with invalid format."""
        test_data = {"test": "data"}

        with self.assertRaises(VisualizationError):
            self.visualizer.export_visualization_data(
                test_data, "test_export", format="invalid"
            )

    @patch.object(Visualizer, 'matplotlib_available', True)
    def test_get_available_styles(self):
        """Test getting available styles."""
        with patch('matplotlib.pyplot.style.available', ['default', 'dark_background', 'seaborn']):
            styles = self.visualizer.get_available_styles()
            self.assertIn('default', styles)
            self.assertIn('dark_background', styles)

    def test_get_available_styles_no_matplotlib(self):
        """Test getting available styles when matplotlib is unavailable."""
        with patch.object(self.visualizer, 'matplotlib_available', False):
            styles = self.visualizer.get_available_styles()
            self.assertEqual(styles, [])

    @patch.object(Visualizer, 'matplotlib_available', True)
    def test_set_style(self):
        """Test setting visualization style."""
        with patch('matplotlib.pyplot.style.use') as mock_use:
            self.visualizer.set_style("dark_background")
            mock_use.assert_called_with("dark_background")
            self.assertEqual(self.visualizer.style, "dark_background")

    def test_set_style_no_matplotlib(self):
        """Test setting style when matplotlib is unavailable."""
        with patch.object(self.visualizer, 'matplotlib_available', False):
            self.visualizer.set_style("dark_background")
            # Should not raise error

    def test_set_style_invalid(self):
        """Test setting invalid style."""
        with patch.object(self.visualizer, 'matplotlib_available', True):
            with patch('matplotlib.pyplot.style.use', side_effect=Exception("Invalid style")):
                with patch('builtins.print'):  # Suppress logging
                    self.visualizer.set_style("invalid_style")
                    # Should fall back to default style
                    self.assertEqual(self.visualizer.style, "default")


if __name__ == '__main__':
    unittest.main()
