"""
Tests for Performance Profiler

Comprehensive tests for the PerformanceProfiler class.
"""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the src directory to the path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model_setup_analysis.performance_profiler import PerformanceProfiler


class TestPerformanceProfiler(unittest.TestCase):
    """Tests for PerformanceProfiler class."""

    def setUp(self):
        """Set up test fixtures."""
        self.profiler = PerformanceProfiler()

    def test_initialization(self):
        """Test PerformanceProfiler initialization."""
        profiler = PerformanceProfiler()
        self.assertEqual(len(profiler._measurements), 0)

    def test_profile_context(self):
        """Test profiling context manager."""
        with self.profiler.profile_context("test_operation"):
            time.sleep(0.01)  # Small delay to ensure measurable time

        # Check that measurement was recorded
        self.assertEqual(len(self.profiler._measurements), 1)

        measurement = self.profiler._measurements[0]
        self.assertEqual(measurement["operation"], "test_operation")
        self.assertIn("duration_seconds", measurement)
        self.assertIn("memory_delta_mb", measurement)
        self.assertIn("peak_memory_mb", measurement)
        self.assertIn("timestamp", measurement)

        # Duration should be very small but positive
        self.assertGreater(measurement["duration_seconds"], 0)
        self.assertLess(measurement["duration_seconds"], 1)  # Less than 1 second

    def test_profile_function(self):
        """Test profiling a function."""
        def test_function(x, y=10):
            time.sleep(0.01)
            return x + y

        result = self.profiler.profile_function(test_function, 5, y=15)

        # Check function result
        self.assertEqual(result, 20)

        # Check that measurement was recorded
        self.assertEqual(len(self.profiler._measurements), 1)
        measurement = self.profiler._measurements[0]
        self.assertEqual(measurement["operation"], "test_function")

    def test_benchmark_model_inference(self):
        """Test benchmarking model inference."""
        def mock_model(input_data):
            time.sleep(0.001)  # Very small delay
            return {"prediction": sum(input_data)}

        input_data = [1, 2, 3, 4, 5]

        results = self.profiler.benchmark_model_inference(
            mock_model, input_data, num_runs=3, warmup_runs=1
        )

        # Check that results contain expected keys
        expected_keys = [
            "num_runs", "mean_inference_time", "std_inference_time",
            "min_inference_time", "max_inference_time", "median_inference_time",
            "mean_memory_delta", "throughput_samples_per_second"
        ]

        for key in expected_keys:
            self.assertIn(key, results)

        # Check values are reasonable
        self.assertEqual(results["num_runs"], 3)
        self.assertGreater(results["mean_inference_time"], 0)
        self.assertGreater(results["throughput_samples_per_second"], 0)

    def test_benchmark_model_inference_failure(self):
        """Test benchmarking when model function fails."""
        def failing_model(input_data):
            raise ValueError("Model failed")

        input_data = [1, 2, 3]

        results = self.profiler.benchmark_model_inference(
            failing_model, input_data, num_runs=2
        )

        # Should return error when no successful runs
        self.assertIn("error", results)
        self.assertEqual(results["error"], "No successful inference runs")

    def test_profile_model_training_step(self):
        """Test profiling training steps."""
        def mock_training_step(batch):
            time.sleep(0.001)
            return 0.5  # Mock loss

        batch_data = {"input": [1, 2, 3], "target": [0, 1, 0]}

        results = self.profiler.profile_model_training_step(
            mock_training_step, batch_data, num_steps=2
        )

        # Check that results contain expected keys
        expected_keys = [
            "num_steps", "mean_step_time", "std_step_time",
            "min_step_time", "max_step_time", "mean_memory_usage",
            "steps_per_second", "profile_timestamp"
        ]

        for key in expected_keys:
            self.assertIn(key, results)

        # Check values are reasonable
        self.assertEqual(results["num_steps"], 2)
        self.assertGreater(results["mean_step_time"], 0)
        self.assertGreater(results["steps_per_second"], 0)

    def test_compare_configurations(self):
        """Test comparing different configurations."""
        def mock_model_factory(config):
            def model(input_data):
                # Simulate different performance based on config
                delay = config.get("delay", 0.001)
                time.sleep(delay)
                return {"result": len(input_data)}
            return model

        configs = [
            {"name": "fast_config", "delay": 0.001},
            {"name": "slow_config", "delay": 0.002}
        ]

        input_data = [1, 2, 3, 4, 5]

        results = self.profiler.compare_configurations(
            configs, mock_model_factory, input_data, num_runs=2
        )

        # Check that results contain both configs
        self.assertIn("fast_config", results)
        self.assertIn("slow_config", results)
        self.assertIn("_summary", results)

        # Check that both configs were successful
        self.assertTrue(results["fast_config"]["success"])
        self.assertTrue(results["slow_config"]["success"])

        # Check summary
        summary = results["_summary"]
        self.assertIn("best_speed_config", summary)
        self.assertIn("best_memory_config", summary)
        self.assertIn("speed_improvement", summary)

    def test_generate_performance_report(self):
        """Test generating performance report."""
        # Add some mock measurements
        self.profiler._measurements = [
            {
                "operation": "test_op_1",
                "duration_seconds": 0.1,
                "memory_delta_mb": 10.5,
                "peak_memory_mb": 100.5,
                "timestamp": time.time()
            },
            {
                "operation": "test_op_2",
                "duration_seconds": 0.2,
                "memory_delta_mb": 15.0,
                "peak_memory_mb": 110.0,
                "timestamp": time.time()
            }
        ]

        report = self.profiler.generate_performance_report()

        # Check that report contains expected sections
        self.assertIn("PERFORMANCE REPORT", report)
        self.assertIn("Total Measurements: 2", report)
        self.assertIn("Timing Statistics:", report)
        self.assertIn("Memory Statistics:", report)
        self.assertIn("Operations Profiled:", report)
        self.assertIn("test_op_1", report)
        self.assertIn("test_op_2", report)

    def test_generate_performance_report_no_measurements(self):
        """Test generating report with no measurements."""
        profiler = PerformanceProfiler()  # Empty profiler

        report = profiler.generate_performance_report()

        self.assertEqual(report, "No performance measurements available")

    def test_clear_measurements(self):
        """Test clearing measurements."""
        # Add some measurements
        self.profiler._measurements = [{"operation": "test"}]

        self.profiler.clear_measurements()

        self.assertEqual(len(self.profiler._measurements), 0)

    def test_save_and_load_measurements(self):
        """Test saving and loading measurements."""
        # Add some mock measurements
        self.profiler._measurements = [
            {
                "operation": "test_op",
                "duration_seconds": 0.1,
                "memory_delta_mb": 10.5,
                "peak_memory_mb": 100.5,
                "timestamp": time.time()
            }
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Save measurements
            self.profiler.save_measurements(temp_path)

            # Clear measurements
            self.profiler.clear_measurements()

            # Load measurements
            self.profiler.load_measurements(temp_path)

            # Check that measurements were loaded
            self.assertEqual(len(self.profiler._measurements), 1)
            self.assertEqual(self.profiler._measurements[0]["operation"], "test_op")

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_load_measurements_nonexistent(self):
        """Test loading measurements from non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.profiler.load_measurements("/nonexistent/path.json")

    def test_get_memory_usage_trend(self):
        """Test analyzing memory usage trends."""
        # Add measurements with increasing memory usage
        base_time = time.time()
        self.profiler._measurements = [
            {
                "operation": "op_1",
                "duration_seconds": 0.1,
                "memory_delta_mb": 10.0,
                "peak_memory_mb": 100.0,
                "timestamp": base_time
            },
            {
                "operation": "op_2",
                "duration_seconds": 0.1,
                "memory_delta_mb": 10.0,
                "peak_memory_mb": 110.0,
                "timestamp": base_time + 1
            },
            {
                "operation": "op_3",
                "duration_seconds": 0.1,
                "memory_delta_mb": 10.0,
                "peak_memory_mb": 120.0,
                "timestamp": base_time + 2
            }
        ]

        trend = self.profiler.get_memory_usage_trend()

        # Check that trend contains expected keys
        expected_keys = [
            "memory_trend_slope", "memory_trend_intercept",
            "memory_increasing", "memory_growth_rate_mb_per_measurement",
            "initial_memory_mb", "final_memory_mb", "total_memory_change_mb",
            "max_memory_mb", "min_memory_mb"
        ]

        for key in expected_keys:
            self.assertIn(key, trend)

        # Check that memory is increasing
        self.assertTrue(trend["memory_increasing"])
        self.assertGreater(trend["memory_trend_slope"], 0)
        self.assertEqual(trend["total_memory_change_mb"], 20.0)

    def test_get_memory_usage_trend_insufficient_data(self):
        """Test memory trend analysis with insufficient data."""
        # Add only one measurement
        self.profiler._measurements = [
            {
                "operation": "op_1",
                "duration_seconds": 0.1,
                "memory_delta_mb": 10.0,
                "peak_memory_mb": 100.0,
                "timestamp": time.time()
            }
        ]

        trend = self.profiler.get_memory_usage_trend()

        self.assertIn("error", trend)
        self.assertEqual(trend["error"], "Need at least 2 measurements for trend analysis")

    def test_get_memory_usage_trend_no_measurements(self):
        """Test memory trend analysis with no measurements."""
        profiler = PerformanceProfiler()  # Empty profiler

        trend = profiler.get_memory_usage_trend()

        self.assertIn("error", trend)
        self.assertEqual(trend["error"], "No measurements available")

    def test_detect_performance_anomalies(self):
        """Test detecting performance anomalies."""
        # Create measurements with one very slow operation
        base_time = time.time()
        self.profiler._measurements = [
            {
                "operation": "normal_op_1",
                "duration_seconds": 0.1,
                "memory_delta_mb": 10.0,
                "peak_memory_mb": 100.0,
                "timestamp": base_time
            },
            {
                "operation": "normal_op_2",
                "duration_seconds": 0.11,
                "memory_delta_mb": 10.0,
                "peak_memory_mb": 100.0,
                "timestamp": base_time + 1
            },
            {
                "operation": "slow_op",  # This should be detected as anomaly
                "duration_seconds": 10.0,  # Much slower (100x the normal 0.1s)
                "memory_delta_mb": 10.0,
                "peak_memory_mb": 100.0,
                "timestamp": base_time + 2
            }
        ]

        anomalies = self.profiler.detect_performance_anomalies(threshold_std=1.0)

        # Should detect the slow operation as an anomaly
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]["measurement"]["operation"], "slow_op")
        self.assertEqual(anomalies[0]["type"], "slow")
        self.assertGreater(anomalies[0]["deviation_sigma"], 1.0)

    def test_detect_performance_anomalies_no_anomalies(self):
        """Test detecting anomalies when all operations are similar."""
        # Create measurements with similar durations
        base_time = time.time()
        self.profiler._measurements = [
            {
                "operation": "op_1",
                "duration_seconds": 0.1,
                "memory_delta_mb": 10.0,
                "peak_memory_mb": 100.0,
                "timestamp": base_time + i
            }
            for i in range(5)  # All similar durations
        ]

        anomalies = self.profiler.detect_performance_anomalies(threshold_std=2.0)

        # Should not detect any anomalies
        self.assertEqual(len(anomalies), 0)

    def test_detect_performance_anomalies_fast_operation(self):
        """Test detecting fast operation as anomaly."""
        # Create measurements with one very fast operation
        base_time = time.time()
        self.profiler._measurements = [
            {
                "operation": "normal_op_1",
                "duration_seconds": 0.1,
                "memory_delta_mb": 10.0,
                "peak_memory_mb": 100.0,
                "timestamp": base_time
            },
            {
                "operation": "fast_op",  # This should be detected as anomaly
                "duration_seconds": 0.001,  # Much faster (100x faster than normal 0.1s)
                "memory_delta_mb": 10.0,
                "peak_memory_mb": 100.0,
                "timestamp": base_time + 1
            },
            {
                "operation": "normal_op_2",
                "duration_seconds": 0.11,
                "memory_delta_mb": 10.0,
                "peak_memory_mb": 100.0,
                "timestamp": base_time + 2
            }
        ]

        anomalies = self.profiler.detect_performance_anomalies(threshold_std=1.0)

        # Should detect the fast operation as an anomaly
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]["measurement"]["operation"], "fast_op")
        self.assertEqual(anomalies[0]["type"], "fast")
        self.assertLess(anomalies[0]["deviation_sigma"], -1.0)  # Negative for fast


if __name__ == '__main__':
    unittest.main()
