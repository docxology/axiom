"""
Tests for Result Analyzer

Comprehensive tests for the ResultAnalyzer class.
"""

import unittest
from pathlib import Path

# Add the src directory to the path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from operation.result_analyzer import ResultAnalyzer, AnalysisSummary
from operation.model_runner import RunResult


class TestAnalysisSummary(unittest.TestCase):
    """Tests for AnalysisSummary dataclass."""

    def test_analysis_summary_creation(self):
        """Test creating an analysis summary."""
        summary = AnalysisSummary(
            total_runs=10,
            successful_runs=8,
            failed_runs=2,
            performance_stats={"mean_reward": 100.0},
            stability_metrics={"consistency": 0.9},
            convergence_analysis={"convergence_detected": True},
            outlier_analysis={"outliers_detected": False}
        )

        self.assertEqual(summary.total_runs, 10)
        self.assertEqual(summary.successful_runs, 8)
        self.assertEqual(summary.failed_runs, 2)
        self.assertIn("mean_reward", summary.performance_stats)
        self.assertIn("consistency", summary.stability_metrics)


class TestResultAnalyzer(unittest.TestCase):
    """Tests for ResultAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ResultAnalyzer()

        # Create sample run results
        self.sample_results = [
            RunResult(
                run_id="run_001",
                model_type="rmm",
                game_name="game1",
                total_steps=1000,
                total_episodes=5,
                total_reward=500.0,
                average_reward=100.0,
                best_reward=150.0,
                worst_reward=50.0,
                duration_seconds=120.0,
                success=True
            ),
            RunResult(
                run_id="run_002",
                model_type="rmm",
                game_name="game1",
                total_steps=1000,
                total_episodes=5,
                total_reward=600.0,
                average_reward=120.0,
                best_reward=180.0,
                worst_reward=60.0,
                duration_seconds=110.0,
                success=True
            ),
            RunResult(
                run_id="run_003",
                model_type="rmm",
                game_name="game1",
                total_steps=1000,
                total_episodes=5,
                total_reward=550.0,
                average_reward=110.0,
                best_reward=160.0,
                worst_reward=55.0,
                duration_seconds=115.0,
                success=True
            ),
            RunResult(
                run_id="run_004",
                model_type="rmm",
                game_name="game1",
                total_steps=0,
                total_episodes=0,
                total_reward=0.0,
                average_reward=0.0,
                best_reward=0.0,
                worst_reward=0.0,
                duration_seconds=5.0,
                success=False,
                error_message="Test failure"
            )
        ]

    def test_initialization(self):
        """Test ResultAnalyzer initialization."""
        analyzer = ResultAnalyzer()
        self.assertIsNotNone(analyzer.logger)

    def test_analyze_run_results_empty(self):
        """Test analyzing empty results."""
        summary = self.analyzer.analyze_run_results([])

        self.assertEqual(summary.total_runs, 0)
        self.assertEqual(summary.successful_runs, 0)
        self.assertEqual(summary.failed_runs, 0)

    def test_analyze_run_results_with_data(self):
        """Test analyzing results with actual data."""
        summary = self.analyzer.analyze_run_results(self.sample_results)

        self.assertEqual(summary.total_runs, 4)
        self.assertEqual(summary.successful_runs, 3)
        self.assertEqual(summary.failed_runs, 1)

        # Check performance stats
        self.assertIn("mean_reward", summary.performance_stats)
        self.assertIn("std_reward", summary.performance_stats)
        self.assertIn("mean_duration", summary.performance_stats)

        # Check stability metrics
        self.assertIn("coefficient_of_variation", summary.stability_metrics)
        self.assertIn("performance_consistency", summary.stability_metrics)

        # Check convergence analysis
        self.assertIn("convergence_detected", summary.convergence_analysis)
        self.assertIn("convergence_rate", summary.convergence_analysis)

        # Check outlier analysis
        self.assertIn("outliers_detected", summary.outlier_analysis)
        self.assertIn("outlier_count", summary.outlier_analysis)

    def test_calculate_performance_stats(self):
        """Test calculating performance statistics."""
        successful_runs = [r for r in self.sample_results if r.success]
        stats = self.analyzer._calculate_performance_stats(successful_runs)

        expected_keys = [
            "mean_reward", "median_reward", "std_reward",
            "min_reward", "max_reward", "mean_duration",
            "mean_steps", "total_runs_analyzed"
        ]

        for key in expected_keys:
            self.assertIn(key, stats)

        self.assertEqual(stats["total_runs_analyzed"], 3)
        self.assertAlmostEqual(stats["mean_reward"], 110.0)
        self.assertEqual(stats["min_reward"], 100.0)
        self.assertEqual(stats["max_reward"], 120.0)

    def test_calculate_stability_metrics(self):
        """Test calculating stability metrics."""
        successful_runs = [r for r in self.sample_results if r.success]
        metrics = self.analyzer._calculate_stability_metrics(successful_runs)

        expected_keys = [
            "coefficient_of_variation", "performance_consistency",
            "reward_range", "stability_score"
        ]

        for key in expected_keys:
            self.assertIn(key, metrics)

        # With our test data, there should be some variation
        self.assertGreater(metrics["coefficient_of_variation"], 0)
        self.assertLess(metrics["coefficient_of_variation"], 1)  # Should be reasonable
        self.assertGreater(metrics["performance_consistency"], 0.5)  # Should be reasonably consistent

    def test_calculate_stability_metrics_single_run(self):
        """Test calculating stability metrics with single run."""
        single_run = [self.sample_results[0]]  # Only one successful run
        metrics = self.analyzer._calculate_stability_metrics(single_run)

        self.assertEqual(metrics["coefficient_of_variation"], 0.0)
        self.assertEqual(metrics["performance_consistency"], 1.0)  # Perfectly consistent
        self.assertEqual(metrics["reward_range"], 0.0)

    def test_calculate_stability_metrics_insufficient_data(self):
        """Test calculating stability metrics with insufficient data."""
        metrics = self.analyzer._calculate_stability_metrics([])

        expected_keys = [
            "coefficient_of_variation", "performance_consistency",
            "reward_range"
        ]

        for key in expected_keys:
            self.assertIn(key, metrics)
            self.assertEqual(metrics[key], 0.0)

    def test_analyze_convergence(self):
        """Test convergence analysis."""
        successful_runs = [r for r in self.sample_results if r.success]
        convergence = self.analyzer._analyze_convergence(successful_runs)

        expected_keys = [
            "convergence_detected", "convergence_rate",
            "stability_threshold", "convergence_step",
            "best_performing_runs"
        ]

        for key in expected_keys:
            self.assertIn(key, convergence)

        # With our test data, convergence should be detected
        self.assertIsInstance(convergence["convergence_detected"], bool)
        self.assertGreater(convergence["convergence_rate"], 0)

    def test_detect_outliers(self):
        """Test outlier detection."""
        successful_runs = [r for r in self.sample_results if r.success]
        outliers = self.analyzer._detect_outliers(successful_runs)

        expected_keys = [
            "outliers_detected", "outlier_count", "outlier_runs",
            "iqr_multiplier", "lower_bound", "upper_bound"
        ]

        for key in expected_keys:
            self.assertIn(key, outliers)

        # With our test data, there should be no outliers (data is fairly consistent)
        self.assertIsInstance(outliers["outliers_detected"], bool)
        self.assertGreaterEqual(outliers["outlier_count"], 0)

    def test_detect_outliers_insufficient_data(self):
        """Test outlier detection with insufficient data."""
        outliers = self.analyzer._detect_outliers([self.sample_results[0]])

        self.assertFalse(outliers["outliers_detected"])
        self.assertEqual(outliers["outlier_count"], 0)
        self.assertEqual(len(outliers["outlier_runs"]), 0)

    def test_compare_configurations_empty(self):
        """Test comparing configurations with empty data."""
        comparison = self.analyzer.compare_configurations({})
        self.assertIn("error", comparison)

    def test_compare_configurations_with_data(self):
        """Test comparing configurations with actual data."""
        results_by_config = {
            "config_a": [r for r in self.sample_results if r.success][:2],  # First 2 successful runs
            "config_b": [r for r in self.sample_results if r.success][2:]   # Last successful run
        }

        comparison = self.analyzer.compare_configurations(results_by_config)

        expected_keys = [
            "configurations_compared", "config_names", "performance_ranking",
            "statistical_significance", "best_configuration", "worst_configuration"
        ]

        for key in expected_keys:
            self.assertIn(key, comparison)

        self.assertEqual(comparison["configurations_compared"], 2)
        self.assertEqual(len(comparison["config_names"]), 2)
        self.assertEqual(len(comparison["performance_ranking"]), 2)
        self.assertIsInstance(comparison["best_configuration"], str)
        self.assertIsInstance(comparison["worst_configuration"], str)

    def test_compare_configurations_different_metric(self):
        """Test comparing configurations with different metric."""
        results_by_config = {
            "config_a": [self.sample_results[0]],  # Lower duration
            "config_b": [self.sample_results[1]]   # Higher duration
        }

        comparison = self.analyzer.compare_configurations(results_by_config, metric="duration")

        self.assertEqual(comparison["configurations_compared"], 2)
        # config_a should be better (lower duration)
        self.assertEqual(comparison["best_configuration"], "config_a")

    def test_analyze_performance_trends_insufficient_data(self):
        """Test analyzing performance trends with insufficient data."""
        trends = self.analyzer.analyze_performance_trends([self.sample_results[0]])

        self.assertFalse(trends["trend_analysis_available"])
        self.assertEqual(trends["trend_slope"], 0.0)
        self.assertEqual(trends["improvement_rate"], 0.0)
        self.assertEqual(trends["convergence_pattern"], "insufficient_data")

    def test_analyze_performance_trends_with_data(self):
        """Test analyzing performance trends with sufficient data."""
        # Create runs with improving performance
        improving_results = [
            RunResult(
                run_id=f"trend_{i}",
                model_type="rmm",
                game_name="game1",
                total_steps=100,
                total_episodes=1,
                total_reward=50.0 + i * 10,  # Improving: 50, 60, 70, 80, 90
                average_reward=50.0 + i * 10,
                best_reward=50.0 + i * 10,
                worst_reward=50.0 + i * 10,
                duration_seconds=10.0,
                success=True
            )
            for i in range(5)
        ]

        trends = self.analyzer.analyze_performance_trends(improving_results)

        self.assertTrue(trends["trend_analysis_available"])
        self.assertGreater(trends["trend_slope"], 0)  # Should show improvement
        self.assertEqual(trends["trend_direction"], "improving")
        self.assertGreater(trends["improvement_rate"], 0)

    def test_generate_performance_report_empty(self):
        """Test generating performance report with no data."""
        report = self.analyzer.generate_performance_report([])
        self.assertIn("No results to analyze", report)

    def test_generate_performance_report_with_data(self):
        """Test generating performance report with data."""
        report = self.analyzer.generate_performance_report(self.sample_results)

        # Check that report contains expected sections
        self.assertIn("PERFORMANCE ANALYSIS", report)
        self.assertIn("EXECUTIVE SUMMARY", report)
        self.assertIn("PERFORMANCE STATISTICS", report)
        self.assertIn("STABILITY ANALYSIS", report)
        self.assertIn("CONVERGENCE ANALYSIS", report)
        self.assertIn("OUTLIER ANALYSIS", report)

        # Check that key metrics are present
        self.assertIn("Total Runs: 4", report)
        self.assertIn("Successful Runs: 3", report)
        self.assertIn("Failed Runs: 1", report)

    def test_calculate_confidence_intervals_insufficient_data(self):
        """Test calculating confidence intervals with insufficient data."""
        intervals = self.analyzer.calculate_confidence_intervals([self.sample_results[0]])

        self.assertFalse(intervals["confidence_intervals_available"])
        self.assertIn("error", intervals)

    def test_calculate_confidence_intervals_with_data(self):
        """Test calculating confidence intervals with sufficient data."""
        successful_runs = [r for r in self.sample_results if r.success]
        intervals = self.analyzer.calculate_confidence_intervals(successful_runs)

        self.assertTrue(intervals["confidence_intervals_available"])

        expected_keys = [
            "mean_reward", "confidence_level", "confidence_interval",
            "margin_of_error", "t_value", "sample_size", "standard_error"
        ]

        for key in expected_keys:
            self.assertIn(key, intervals)

        self.assertEqual(intervals["confidence_level"], 0.95)
        self.assertEqual(intervals["sample_size"], 3)

        # Check confidence interval structure
        ci = intervals["confidence_interval"]
        self.assertEqual(len(ci), 2)
        self.assertLess(ci[0], ci[1])  # Lower bound < upper bound


if __name__ == '__main__':
    unittest.main()
