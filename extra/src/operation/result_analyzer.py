"""
Result Analyzer for AXIOM Extensions

Provides comprehensive analysis tools for experimental results,
performance metrics, and statistical comparisons.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np
import statistics
from dataclasses import dataclass

from .model_runner import RunResult

logger = logging.getLogger(__name__)


@dataclass
class AnalysisSummary:
    """Summary of analysis results."""
    total_runs: int
    successful_runs: int
    failed_runs: int
    performance_stats: Dict[str, float]
    stability_metrics: Dict[str, float]
    convergence_analysis: Dict[str, Any]
    outlier_analysis: Dict[str, Any]


class ResultAnalyzer:
    """
    Analyzes experimental results and provides statistical insights.

    This class provides methods to:
    - Analyze performance distributions
    - Detect statistical significance
    - Identify performance trends
    - Generate comparative analyses
    """

    def __init__(self):
        """Initialize result analyzer."""
        self.logger = logging.getLogger(__name__)

    def analyze_run_results(self, results: List[RunResult]) -> AnalysisSummary:
        """
        Perform comprehensive analysis of run results.

        Args:
            results: List of run results to analyze

        Returns:
            AnalysisSummary: Comprehensive analysis results
        """
        if not results:
            return AnalysisSummary(
                total_runs=0,
                successful_runs=0,
                failed_runs=0,
                performance_stats={},
                stability_metrics={},
                convergence_analysis={},
                outlier_analysis={}
            )

        successful_runs = [r for r in results if r.success]
        failed_runs = [r for r in results if not r.success]

        performance_stats = self._calculate_performance_stats(successful_runs)
        stability_metrics = self._calculate_stability_metrics(successful_runs)
        convergence_analysis = self._analyze_convergence(successful_runs)
        outlier_analysis = self._detect_outliers(successful_runs)

        return AnalysisSummary(
            total_runs=len(results),
            successful_runs=len(successful_runs),
            failed_runs=len(failed_runs),
            performance_stats=performance_stats,
            stability_metrics=stability_metrics,
            convergence_analysis=convergence_analysis,
            outlier_analysis=outlier_analysis
        )

    def _calculate_performance_stats(self, successful_runs: List[RunResult]) -> Dict[str, float]:
        """Calculate basic performance statistics."""
        if not successful_runs:
            return {
                "mean_reward": 0.0,
                "median_reward": 0.0,
                "std_reward": 0.0,
                "min_reward": 0.0,
                "max_reward": 0.0,
                "mean_duration": 0.0,
                "mean_steps": 0.0
            }

        rewards = [r.average_reward for r in successful_runs]
        durations = [r.duration_seconds for r in successful_runs]
        steps = [r.total_steps for r in successful_runs]

        return {
            "mean_reward": statistics.mean(rewards),
            "median_reward": statistics.median(rewards),
            "std_reward": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
            "min_reward": min(rewards),
            "max_reward": max(rewards),
            "mean_duration": statistics.mean(durations),
            "mean_steps": statistics.mean(steps),
            "total_runs_analyzed": len(successful_runs)
        }

    def _calculate_stability_metrics(self, successful_runs: List[RunResult]) -> Dict[str, float]:
        """Calculate stability metrics."""
        if not successful_runs:
            return {
                "coefficient_of_variation": 0.0,
                "performance_consistency": 0.0,
                "reward_range": 0.0
            }

        rewards = [r.average_reward for r in successful_runs]

        if len(rewards) < 2:
            return {
                "coefficient_of_variation": 0.0,
                "performance_consistency": 1.0,  # Perfectly consistent with one run
                "reward_range": 0.0
            }

        mean_reward = statistics.mean(rewards)
        std_reward = statistics.stdev(rewards)

        coefficient_of_variation = std_reward / abs(mean_reward) if mean_reward != 0 else 0.0
        performance_consistency = 1.0 / (1.0 + coefficient_of_variation)  # Higher is more consistent
        reward_range = max(rewards) - min(rewards)

        return {
            "coefficient_of_variation": coefficient_of_variation,
            "performance_consistency": performance_consistency,
            "reward_range": reward_range,
            "stability_score": performance_consistency  # Alias for backward compatibility
        }

    def _analyze_convergence(self, successful_runs: List[RunResult]) -> Dict[str, Any]:
        """Analyze convergence patterns."""
        if not successful_runs:
            return {
                "convergence_detected": False,
                "convergence_rate": 0.0,
                "stability_threshold": 0.0,
                "convergence_step": 0
            }

        # Sort by performance for convergence analysis
        sorted_runs = sorted(successful_runs, key=lambda r: r.average_reward, reverse=True)

        # Simple convergence detection: check if top performers have similar rewards
        if len(sorted_runs) >= 3:
            top_rewards = [r.average_reward for r in sorted_runs[:3]]
            convergence_detected = statistics.stdev(top_rewards) < 0.1 * statistics.mean(top_rewards)
        else:
            convergence_detected = True

        return {
            "convergence_detected": convergence_detected,
            "convergence_rate": len([r for r in successful_runs if r.average_reward > 0]) / len(successful_runs),
            "stability_threshold": 0.1,  # 10% variation threshold
            "convergence_step": len(successful_runs) // 2 if len(successful_runs) > 1 else 0,
            "best_performing_runs": len([r for r in successful_runs if r.average_reward >= sorted_runs[0].average_reward * 0.9])
        }

    def _detect_outliers(self, successful_runs: List[RunResult]) -> Dict[str, Any]:
        """Detect performance outliers."""
        if len(successful_runs) < 3:
            return {
                "outliers_detected": False,
                "outlier_count": 0,
                "outlier_runs": [],
                "iqr_multiplier": 1.5
            }

        rewards = [r.average_reward for r in successful_runs]

        # Use IQR method for outlier detection
        q1 = np.percentile(rewards, 25)
        q3 = np.percentile(rewards, 75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = []
        for run in successful_runs:
            if run.average_reward < lower_bound or run.average_reward > upper_bound:
                outliers.append({
                    "run_id": run.run_id,
                    "reward": run.average_reward,
                    "deviation": "high" if run.average_reward > upper_bound else "low"
                })

        return {
            "outliers_detected": len(outliers) > 0,
            "outlier_count": len(outliers),
            "outlier_runs": outliers,
            "iqr_multiplier": 1.5,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }

    def compare_configurations(
        self,
        results_by_config: Dict[str, List[RunResult]],
        metric: str = "average_reward"
    ) -> Dict[str, Any]:
        """
        Compare performance across different configurations.

        Args:
            results_by_config: Results grouped by configuration
            metric: Metric to compare (default: average_reward)

        Returns:
            Dictionary with comparison results
        """
        if not results_by_config:
            return {"error": "No results to compare"}

        comparison = {
            "configurations_compared": len(results_by_config),
            "config_names": list(results_by_config.keys()),
            "performance_ranking": [],
            "statistical_significance": {},
            "best_configuration": None,
            "worst_configuration": None
        }

        # Calculate average performance for each configuration
        config_performance = {}
        for config_name, results in results_by_config.items():
            successful_runs = [r for r in results if r.success]
            if successful_runs:
                if metric == "average_reward":
                    performance = [r.average_reward for r in successful_runs]
                elif metric == "best_reward":
                    performance = [r.best_reward for r in successful_runs]
                elif metric == "duration":
                    performance = [r.duration_seconds for r in successful_runs]
                else:
                    performance = [getattr(r, metric, 0) for r in successful_runs]

                config_performance[config_name] = {
                    "mean": statistics.mean(performance),
                    "std": statistics.stdev(performance) if len(performance) > 1 else 0.0,
                    "count": len(performance),
                    "values": performance
                }

        if config_performance:
            # Sort by mean performance
            sorted_configs = sorted(config_performance.items(), key=lambda x: x[1]["mean"], reverse=True)
            comparison["performance_ranking"] = [
                {
                    "config": name,
                    "mean_performance": data["mean"],
                    "std_performance": data["std"],
                    "rank": i + 1
                }
                for i, (name, data) in enumerate(sorted_configs)
            ]

            comparison["best_configuration"] = sorted_configs[0][0]
            comparison["worst_configuration"] = sorted_configs[-1][0]

            # Simple statistical significance test (t-test approximation)
            if len(sorted_configs) >= 2:
                best_config = sorted_configs[0][1]
                second_config = sorted_configs[1][1]

                # Calculate t-statistic approximation
                if best_config["std"] > 0 and second_config["std"] > 0:
                    t_stat = abs(best_config["mean"] - second_config["mean"]) / \
                            ((best_config["std"]**2 / best_config["count"] + second_config["std"]**2 / second_config["count"])**0.5)
                    comparison["statistical_significance"]["t_statistic"] = t_stat
                    comparison["statistical_significance"]["significant_at_95"] = t_stat > 1.96

        return comparison

    def analyze_performance_trends(self, results: List[RunResult]) -> Dict[str, Any]:
        """
        Analyze performance trends over time or runs.

        Args:
            results: List of run results (assumed to be in chronological order)

        Returns:
            Dictionary with trend analysis
        """
        successful_runs = [r for r in results if r.success]

        if len(successful_runs) < 3:
            return {
                "trend_analysis_available": False,
                "trend_slope": 0.0,
                "improvement_rate": 0.0,
                "convergence_pattern": "insufficient_data"
            }

        # Calculate trend using linear regression
        rewards = [r.average_reward for r in successful_runs]
        x = np.arange(len(rewards))

        # Linear regression
        slope, intercept = np.polyfit(x, rewards, 1)
        trend_correlation = np.corrcoef(x, rewards)[0, 1]

        # Determine trend direction
        if slope > 0.01:
            trend_direction = "improving"
        elif slope < -0.01:
            trend_direction = "declining"
        else:
            trend_direction = "stable"

        # Calculate improvement rate
        first_half = rewards[:len(rewards)//2]
        second_half = rewards[len(rewards)//2:]

        if first_half and second_half:
            first_avg = statistics.mean(first_half)
            second_avg = statistics.mean(second_half)
            improvement_rate = (second_avg - first_avg) / abs(first_avg) if first_avg != 0 else 0.0
        else:
            improvement_rate = 0.0

        return {
            "trend_analysis_available": True,
            "trend_slope": slope,
            "trend_correlation": trend_correlation,
            "trend_direction": trend_direction,
            "improvement_rate": improvement_rate,
            "convergence_pattern": "converging" if abs(slope) < 0.01 else "diverging",
            "data_points": len(rewards),
            "first_half_avg": statistics.mean(first_half) if first_half else 0.0,
            "second_half_avg": statistics.mean(second_half) if second_half else 0.0
        }

    def generate_performance_report(self, results: List[RunResult], title: str = "Performance Analysis") -> str:
        """
        Generate a comprehensive performance report.

        Args:
            results: List of run results to analyze
            title: Title for the report

        Returns:
            Formatted performance report
        """
        analysis = self.analyze_run_results(results)

        if analysis.total_runs == 0:
            return f"{title}\n{'=' * len(title)}\nNo results to analyze"

        report = f"""
PERFORMANCE ANALYSIS
====================

EXECUTIVE SUMMARY
================
Total Runs: {analysis.total_runs}
Successful Runs: {analysis.successful_runs}
Failed Runs: {analysis.failed_runs}
Success Rate: {analysis.successful_runs / analysis.total_runs * 100:.1f}%

PERFORMANCE STATISTICS
======================
"""

        if analysis.performance_stats:
            stats = analysis.performance_stats
            report += f"""
Mean Reward: {stats['mean_reward']:.3f}
Median Reward: {stats['median_reward']:.3f}
Std Deviation: {stats['std_reward']:.3f}
Min/Max Reward: {stats['min_reward']:.3f} / {stats['max_reward']:.3f}
Mean Duration: {stats['mean_duration']:.2f}s
Mean Steps: {stats['mean_steps']:.0f}
"""

        report += f"""
STABILITY ANALYSIS
==================
"""

        if analysis.stability_metrics:
            stability = analysis.stability_metrics
            report += f"""
Coefficient of Variation: {stability['coefficient_of_variation']:.3f}
Performance Consistency: {stability['performance_consistency']:.3f}
Reward Range: {stability['reward_range']:.3f}
Stability Score: {stability['stability_score']:.3f}
"""

        report += f"""
CONVERGENCE ANALYSIS
====================
"""

        if analysis.convergence_analysis:
            conv = analysis.convergence_analysis
            report += f"""
Convergence Detected: {conv['convergence_detected']}
Convergence Rate: {conv['convergence_rate']:.3f}
Best Performing Runs: {conv['best_performing_runs']}
"""

        report += f"""
OUTLIER ANALYSIS
================
"""

        if analysis.outlier_analysis:
            outliers = analysis.outlier_analysis
            report += f"""
Outliers Detected: {outliers['outliers_detected']}
Outlier Count: {outliers['outlier_count']}
Outlier Bounds: [{outliers['lower_bound']:.3f}, {outliers['upper_bound']:.3f}]
"""

        return report.strip()

    def calculate_confidence_intervals(self, results: List[RunResult], confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Calculate confidence intervals for performance metrics.

        Args:
            results: List of run results
            confidence_level: Confidence level (default: 0.95)

        Returns:
            Dictionary with confidence interval analysis
        """
        successful_runs = [r for r in results if r.success]

        if len(successful_runs) < 2:
            return {
                "confidence_intervals_available": False,
                "error": "Need at least 2 successful runs for confidence intervals"
            }

        rewards = [r.average_reward for r in successful_runs]

        # Calculate confidence interval using t-distribution
        mean_reward = statistics.mean(rewards)
        std_reward = statistics.stdev(rewards)
        n = len(rewards)

        # t-value for 95% confidence with n-1 degrees of freedom
        t_value = 2.776 if n > 30 else [0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228][min(n-1, 10)]

        margin_of_error = t_value * (std_reward / (n ** 0.5))
        confidence_interval = (mean_reward - margin_of_error, mean_reward + margin_of_error)

        return {
            "confidence_intervals_available": True,
            "mean_reward": mean_reward,
            "confidence_level": confidence_level,
            "confidence_interval": confidence_interval,
            "margin_of_error": margin_of_error,
            "t_value": t_value,
            "sample_size": n,
            "standard_error": std_reward / (n ** 0.5)
        }
