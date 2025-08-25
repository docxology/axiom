"""
Performance Profiler for AXIOM Extensions

Provides performance profiling and benchmarking tools for AXIOM models,
helping optimize configurations and identify bottlenecks.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from contextlib import contextmanager
import psutil
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """
    Profiles performance characteristics of AXIOM models.

    This class provides methods to:
    - Profile model inference time
    - Monitor memory usage
    - Benchmark different configurations
    - Generate performance reports
    """

    def __init__(self):
        """Initialize performance profiler."""
        self.logger = logging.getLogger(__name__)
        self._measurements: List[Dict[str, Any]] = []

    @contextmanager
    def profile_context(self, operation_name: str):
        """
        Context manager for profiling operations.

        Args:
            operation_name: Name of the operation being profiled
        """
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            measurement = {
                "operation": operation_name,
                "duration_seconds": end_time - start_time,
                "memory_delta_mb": end_memory - start_memory,
                "peak_memory_mb": end_memory,
                "timestamp": time.time()
            }

            self._measurements.append(measurement)
            self.logger.info(f"Profiled {operation_name}: {measurement['duration_seconds']:.3f}s, "
                           f"Δmemory: {measurement['memory_delta_mb']:.1f}MB")

    def profile_function(self, func: Callable, *args, operation_name: Optional[str] = None, **kwargs) -> Any:
        """
        Profile a function execution.

        Args:
            func: Function to profile
            *args: Arguments to pass to function
            operation_name: Optional name for the operation
            **kwargs: Keyword arguments to pass to function

        Returns:
            Result of the function call
        """
        if operation_name is None:
            operation_name = f"{func.__name__}"

        with self.profile_context(operation_name):
            return func(*args, **kwargs)

    def benchmark_model_inference(
        self,
        model_func: Callable,
        input_data: Any,
        num_runs: int = 10,
        warmup_runs: int = 3
    ) -> Dict[str, Any]:
        """
        Benchmark model inference performance.

        Args:
            model_func: Model inference function
            input_data: Input data for inference
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs

        Returns:
            Dictionary with benchmark results
        """
        # Warmup runs
        for _ in range(warmup_runs):
            try:
                model_func(input_data)
            except Exception as e:
                self.logger.warning(f"Warmup failed: {e}")

        # Benchmark runs
        inference_times = []
        memory_usage = []

        for i in range(num_runs):
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            with self.profile_context(f"inference_run_{i}"):
                try:
                    result = model_func(input_data)
                except Exception as e:
                    self.logger.error(f"Inference run {i} failed: {e}")
                    continue

            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_usage.append(end_memory - start_memory)

            # Get the last measurement (current inference)
            if self._measurements:
                inference_times.append(self._measurements[-1]["duration_seconds"])

        # Calculate statistics
        if not inference_times:
            return {"error": "No successful inference runs"}

        results = {
            "num_runs": len(inference_times),
            "mean_inference_time": np.mean(inference_times),
            "std_inference_time": np.std(inference_times),
            "min_inference_time": np.min(inference_times),
            "max_inference_time": np.max(inference_times),
            "median_inference_time": np.median(inference_times),
            "mean_memory_delta": np.mean(memory_usage) if memory_usage else 0,
            "memory_std": np.std(memory_usage) if memory_usage else 0,
            "throughput_samples_per_second": 1.0 / np.mean(inference_times) if inference_times else 0,
            "benchmark_timestamp": time.time()
        }

        return results

    def profile_model_training_step(
        self,
        training_func: Callable,
        batch_data: Any,
        num_steps: int = 10
    ) -> Dict[str, Any]:
        """
        Profile model training steps.

        Args:
            training_func: Training step function
            batch_data: Batch data for training
            num_steps: Number of training steps to profile

        Returns:
            Dictionary with training profile results
        """
        step_times = []
        memory_usage = []
        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        for step in range(num_steps):
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            with self.profile_context(f"training_step_{step}"):
                try:
                    loss = training_func(batch_data)
                except Exception as e:
                    self.logger.error(f"Training step {step} failed: {e}")
                    continue

            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_usage.append(end_memory - baseline_memory)

            if self._measurements:
                step_times.append(self._measurements[-1]["duration_seconds"])

        if not step_times:
            return {"error": "No successful training steps"}

        results = {
            "num_steps": len(step_times),
            "mean_step_time": np.mean(step_times),
            "std_step_time": np.std(step_times),
            "min_step_time": np.min(step_times),
            "max_step_time": np.max(step_times),
            "mean_memory_usage": np.mean(memory_usage) if memory_usage else 0,
            "max_memory_usage": np.max(memory_usage) if memory_usage else 0,
            "memory_growth_rate": (
                (memory_usage[-1] - memory_usage[0]) / len(memory_usage)
                if len(memory_usage) > 1 else 0
            ),
            "steps_per_second": 1.0 / np.mean(step_times) if step_times else 0,
            "profile_timestamp": time.time()
        }

        return results

    def compare_configurations(
        self,
        configs: List[Dict[str, Any]],
        model_func: Callable,
        input_data: Any,
        num_runs: int = 5
    ) -> Dict[str, Any]:
        """
        Compare performance of different model configurations.

        Args:
            configs: List of model configurations to compare
            model_func: Function that takes config and returns model
            input_data: Input data for testing
            num_runs: Number of runs per configuration

        Returns:
            Dictionary with comparison results
        """
        comparison_results = {}

        for i, config in enumerate(configs):
            config_name = config.get('name', f'config_{i}')

            try:
                # Create model with configuration
                model = model_func(config)

                # Profile inference
                inference_results = self.benchmark_model_inference(
                    model, input_data, num_runs=num_runs
                )

                comparison_results[config_name] = {
                    "config": config,
                    "inference_results": inference_results,
                    "success": True
                }

            except Exception as e:
                self.logger.error(f"Failed to profile config {config_name}: {e}")
                comparison_results[config_name] = {
                    "config": config,
                    "error": str(e),
                    "success": False
                }

        # Generate comparison summary
        successful_configs = [name for name, result in comparison_results.items() if result["success"]]

        if len(successful_configs) > 1:
            best_speed = min(successful_configs,
                           key=lambda x: comparison_results[x]["inference_results"]["mean_inference_time"])
            best_memory = min(successful_configs,
                            key=lambda x: comparison_results[x]["inference_results"]["mean_memory_delta"])

            comparison_results["_summary"] = {
                "best_speed_config": best_speed,
                "best_memory_config": best_memory,
                "speed_improvement": (
                    comparison_results[successful_configs[0]]["inference_results"]["mean_inference_time"] /
                    comparison_results[best_speed]["inference_results"]["mean_inference_time"]
                ),
                "total_configs_tested": len(configs),
                "successful_configs": len(successful_configs)
            }

        return comparison_results

    def generate_performance_report(self, measurements: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate a comprehensive performance report.

        Args:
            measurements: Optional measurements to include (uses internal if None)

        Returns:
            Formatted performance report
        """
        if measurements is None:
            measurements = self._measurements

        if not measurements:
            return "No performance measurements available"

        # Calculate statistics
        durations = [m["duration_seconds"] for m in measurements]
        memory_deltas = [m["memory_delta_mb"] for m in measurements]

        report = ".3f"".1f"".1f"".1f"".1f"".3f"f"""
PERFORMANCE REPORT
==================

Total Measurements: {len(measurements)}
Total Duration: {sum(durations):.3f} seconds

Timing Statistics:
- Mean Duration: {np.mean(durations):.3f} seconds
- Std Duration: {np.std(durations):.3f} seconds
- Min Duration: {np.min(durations):.3f} seconds
- Max Duration: {np.max(durations):.3f} seconds
- Median Duration: {np.median(durations):.3f} seconds

Memory Statistics:
- Mean Memory Delta: {np.mean(memory_deltas):.1f} MB
- Std Memory Delta: {np.std(memory_deltas):.1f} MB
- Min Memory Delta: {np.min(memory_deltas):.1f} MB
- Max Memory Delta: {np.max(memory_deltas):.1f} MB

Operations Profiled:
"""

        # Group by operation
        operations = {}
        for measurement in measurements:
            op = measurement["operation"]
            if op not in operations:
                operations[op] = []
            operations[op].append(measurement)

        for op_name, op_measurements in operations.items():
            op_durations = [m["duration_seconds"] for m in op_measurements]
            op_memory = [m["memory_delta_mb"] for m in op_measurements]

            report += ".3f"".1f"".3f"".1f"f"""
  {op_name} ({len(op_measurements)} calls):
    - Mean Duration: {np.mean(op_durations):.3f}s
    - Mean Memory: {np.mean(op_memory):.1f}MB
    - Total Time: {sum(op_durations):.3f}s
    - Total Memory: {sum(op_memory):.1f}MB"""

        return report.strip()

    def clear_measurements(self):
        """Clear all stored measurements."""
        self._measurements.clear()
        self.logger.info("Cleared all performance measurements")

    def save_measurements(self, filepath: Union[str, Path]):
        """
        Save measurements to file.

        Args:
            filepath: Path to save measurements
        """
        import json

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self._measurements, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved {len(self._measurements)} measurements to {filepath}")

    def load_measurements(self, filepath: Union[str, Path]):
        """
        Load measurements from file.

        Args:
            filepath: Path to load measurements from
        """
        import json

        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Measurements file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            self._measurements = json.load(f)

        self.logger.info(f"Loaded {len(self._measurements)} measurements from {filepath}")

    def get_memory_usage_trend(self) -> Dict[str, Any]:
        """
        Analyze memory usage trends from measurements.

        Returns:
            Dictionary with memory usage trend analysis
        """
        if not self._measurements:
            return {"error": "No measurements available"}

        memory_values = [m["peak_memory_mb"] for m in self._measurements]

        if len(memory_values) < 2:
            return {"error": "Need at least 2 measurements for trend analysis"}

        # Calculate trend
        x = np.arange(len(memory_values))
        slope, intercept = np.polyfit(x, memory_values, 1)

        trend = {
            "memory_trend_slope": slope,
            "memory_trend_intercept": intercept,
            "memory_increasing": slope > 0,
            "memory_growth_rate_mb_per_measurement": slope,
            "initial_memory_mb": memory_values[0],
            "final_memory_mb": memory_values[-1],
            "total_memory_change_mb": memory_values[-1] - memory_values[0],
            "max_memory_mb": max(memory_values),
            "min_memory_mb": min(memory_values)
        }

        return trend

    def detect_performance_anomalies(self, threshold_std: float = 2.0) -> List[Dict[str, Any]]:
        """
        Detect performance anomalies in measurements.

        Args:
            threshold_std: Standard deviation threshold for anomaly detection

        Returns:
            List of detected anomalies
        """
        if not self._measurements:
            return []

        durations = np.array([m["duration_seconds"] for m in self._measurements])
        mean_duration = np.mean(durations)
        std_duration = np.std(durations)

        anomalies = []

        for i, measurement in enumerate(self._measurements):
            if abs(measurement["duration_seconds"] - mean_duration) > threshold_std * std_duration:
                anomalies.append({
                    "index": i,
                    "measurement": measurement,
                    "deviation_sigma": (measurement["duration_seconds"] - mean_duration) / std_duration,
                    "type": "slow" if measurement["duration_seconds"] > mean_duration else "fast"
                })

        return anomalies
