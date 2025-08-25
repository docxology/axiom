"""
Tests for Plot Manager

Comprehensive tests for the PlotManager class.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the src directory to the path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visualization.plot_manager import PlotManager
from visualization.visualizer import VisualizationError


class TestPlotManager(unittest.TestCase):
    """Tests for PlotManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.plot_manager = PlotManager(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test PlotManager initialization."""
        plot_manager = PlotManager("/tmp/test_plots")
        self.assertIsNotNone(plot_manager.output_dir)
        self.assertIsNotNone(plot_manager.visualizer)
        self.assertIsNotNone(plot_manager.animator)

    @patch('visualization.plot_manager.PlotManager._create_performance_comparison')
    @patch('visualization.plot_manager.PlotManager._create_training_progress')
    @patch('visualization.plot_manager.PlotManager._create_summary_dashboard')
    @patch('visualization.plot_manager.PlotManager._export_visualization_data')
    @patch('visualization.plot_manager.PlotManager._generate_html_report')
    def test_create_experiment_visualization_suite(self, mock_html, mock_export, mock_dashboard, mock_progress, mock_comparison):
        """Test creating complete visualization suite."""
        # Mock return values
        mock_comparison.return_value = "/tmp/plot1.png"
        mock_progress.return_value = "/tmp/plot2.png"
        mock_dashboard.return_value = "/tmp/plot3.png"
        mock_export.return_value = "/tmp/data.json"
        mock_html.return_value = "/tmp/report.html"

        experiment_data = {
            "run_results": [{"run_id": "test", "average_reward": 100.0}],
            "training_history": [{"step": 0, "reward": 50.0}],
            "total_runs": 5,
            "successful_runs": 4,
            "duration_seconds": 120.0
        }

        result = self.plot_manager.create_experiment_visualization_suite(
            experiment_data, "test_experiment", include_animations=False
        )

        self.assertIn("static_plots", result)
        self.assertIn("animations", result)
        self.assertIn("reports", result)
        self.assertIn("data_exports", result)

        # Verify that methods were called
        mock_comparison.assert_called_once()
        mock_progress.assert_called_once()
        mock_dashboard.assert_called_once()
        mock_export.assert_called_once()
        mock_html.assert_called_once()

    def test_create_experiment_visualization_suite_error_handling(self):
        """Test error handling in visualization suite creation."""
        with patch.object(self.plot_manager, '_create_performance_comparison', side_effect=Exception("Test error")):
            experiment_data = {
                "run_results": [{"run_id": "test", "average_reward": 100.0}]
            }

            with self.assertRaises(VisualizationError):
                self.plot_manager.create_experiment_visualization_suite(
                    experiment_data, "test_experiment"
                )

    def test_batch_create_visualizations(self):
        """Test batch creation of visualizations."""
        experiments_data = [
            {"run_results": [{"run_id": "exp1", "average_reward": 100.0}]},
            {"run_results": [{"run_id": "exp2", "average_reward": 120.0}]}
        ]
        experiment_names = ["experiment_1", "experiment_2"]

        with patch.object(self.plot_manager, 'create_experiment_visualization_suite') as mock_suite:
            mock_suite.side_effect = [
                {"static_plots": ["plot1.png"]},
                {"static_plots": ["plot2.png"]}
            ]

            result = self.plot_manager.batch_create_visualizations(
                experiments_data, experiment_names, include_animations=False
            )

            self.assertEqual(len(result), 2)
            self.assertIn("experiment_1", result)
            self.assertIn("experiment_2", result)
            self.assertEqual(mock_suite.call_count, 2)

    def test_batch_create_visualizations_with_errors(self):
        """Test batch creation with some failures."""
        experiments_data = [
            {"run_results": [{"run_id": "exp1", "average_reward": 100.0}]},
            {"run_results": [{"run_id": "exp2", "average_reward": 120.0}]}
        ]
        experiment_names = ["experiment_1", "experiment_2"]

        with patch.object(self.plot_manager, 'create_experiment_visualization_suite') as mock_suite:
            mock_suite.side_effect = [
                {"static_plots": ["plot1.png"]},
                Exception("Failed to create suite")
            ]

            result = self.plot_manager.batch_create_visualizations(
                experiments_data, experiment_names, include_animations=False
            )

            self.assertEqual(len(result), 2)
            self.assertIn("experiment_1", result)
            self.assertIn("experiment_2", result)
            self.assertIn("error", result["experiment_2"])

    def test_get_visualization_statistics(self):
        """Test getting visualization statistics."""
        # Create some dummy files
        suite_dir = self.plot_manager.output_dir / "test_experiment_suite"
        suite_dir.mkdir()

        static_dir = suite_dir / "static"
        static_dir.mkdir()
        (static_dir / "plot1.png").write_text("dummy")
        (static_dir / "plot2.jpg").write_text("dummy")

        anim_dir = suite_dir / "animations"
        anim_dir.mkdir()
        (anim_dir / "anim1.mp4").write_text("dummy")

        report_dir = suite_dir / "reports"
        report_dir.mkdir()
        (report_dir / "report.html").write_text("dummy")

        stats = self.plot_manager.get_visualization_statistics()

        self.assertEqual(stats["static_plots"], 2)
        self.assertEqual(stats["animations"], 1)
        self.assertEqual(stats["reports"], 1)
        self.assertEqual(stats["total_files"], 4)
        self.assertIn("test_experiment", stats["by_experiment"])

    def test_get_visualization_statistics_empty(self):
        """Test getting statistics when no visualizations exist."""
        stats = self.plot_manager.get_visualization_statistics()

        self.assertEqual(stats["total_files"], 0)
        self.assertEqual(stats["static_plots"], 0)
        self.assertEqual(stats["animations"], 0)
        self.assertEqual(stats["reports"], 0)
        self.assertEqual(stats["total_size_mb"], 0)

    def test_cleanup_old_visualizations(self):
        """Test cleaning up old visualization files."""
        import time
        import os

        # Create a dummy file
        test_file = self.plot_manager.output_dir / "old_file.png"
        test_file.write_text("dummy")

        # Set file modification time to 60 days ago
        old_time = time.time() - (60 * 24 * 60 * 60)
        os.utime(test_file, (old_time, old_time))

        # Clean up files older than 30 days
        removed_count = self.plot_manager.cleanup_old_visualizations(days_old=30)

        self.assertEqual(removed_count, 1)
        self.assertFalse(test_file.exists())

    def test_cleanup_old_visualizations_no_files(self):
        """Test cleaning up when no old files exist."""
        removed_count = self.plot_manager.cleanup_old_visualizations(days_old=1)
        self.assertEqual(removed_count, 0)

    @patch('visualization.visualizer.Visualizer.plot_performance_comparison')
    def test_create_performance_comparison(self, mock_plot):
        """Test creating performance comparison plot."""
        mock_plot.return_value = "/tmp/test_plot.png"

        run_results = [{"run_id": "test", "average_reward": 100.0}]

        result = self.plot_manager._create_performance_comparison(run_results, Path("/tmp"), "test")

        self.assertEqual(result, "/tmp/test_plot.png")
        mock_plot.assert_called_once()

    @patch('visualization.visualizer.Visualizer.plot_training_progress')
    def test_create_training_progress(self, mock_plot):
        """Test creating training progress plot."""
        mock_plot.return_value = "/tmp/test_plot.png"

        training_history = [{"step": 0, "reward": 50.0}]

        result = self.plot_manager._create_training_progress(training_history, Path("/tmp"), "test")

        self.assertEqual(result, "/tmp/test_plot.png")
        mock_plot.assert_called_once()

    @patch('visualization.visualizer.Visualizer.plot_model_comparison_matrix')
    def test_create_model_comparison(self, mock_plot):
        """Test creating model comparison matrix."""
        mock_plot.return_value = "/tmp/test_plot.png"

        comparison_data = {"model1": {"accuracy": 0.9}, "model2": {"accuracy": 0.8}}

        result = self.plot_manager._create_model_comparison(comparison_data, Path("/tmp"), "test")

        self.assertEqual(result, "/tmp/test_plot.png")
        mock_plot.assert_called_once()

    @patch('visualization.visualizer.Visualizer.create_summary_dashboard')
    def test_create_summary_dashboard(self, mock_dashboard):
        """Test creating summary dashboard."""
        mock_dashboard.return_value = "/tmp/test_dashboard.png"

        experiment_data = {"performance": {"mean": 100.0}}

        result = self.plot_manager._create_summary_dashboard(experiment_data, Path("/tmp"), "test")

        self.assertEqual(result, "/tmp/test_dashboard.png")
        mock_dashboard.assert_called_once()

    @patch('visualization.visualizer.Visualizer.export_visualization_data')
    def test_export_visualization_data(self, mock_export):
        """Test exporting visualization data."""
        mock_export.return_value = "/tmp/test_data.json"

        experiment_data = {"test": "data"}

        result = self.plot_manager._export_visualization_data(experiment_data, Path("/tmp"), "test")

        self.assertEqual(result, "/tmp/test_data.json")
        mock_export.assert_called_once()


if __name__ == '__main__':
    unittest.main()
