"""
Output Configuration for AXIOM Extensions

Centralized configuration system for managing output directories and paths
across all AXIOM extension modules.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class OutputPaths:
    """Configuration for all output paths used by AXIOM extensions."""

    # Base output directory
    base_dir: Path = field(default_factory=lambda: Path("output"))

    # Environment setup outputs
    environment_configs: Path = field(init=False)
    environment_reports: Path = field(init=False)
    environment_logs: Path = field(init=False)

    # Model setup analysis outputs
    model_configs: Path = field(init=False)
    model_analysis: Path = field(init=False)
    model_performance: Path = field(init=False)

    # Operation outputs
    experiments: Path = field(init=False)
    runs: Path = field(init=False)
    experiment_results: Path = field(init=False)
    run_checkpoints: Path = field(init=False)
    run_results: Path = field(init=False)
    run_videos: Path = field(init=False)

    # Visualization outputs
    visualizations: Path = field(init=False)
    static_plots: Path = field(init=False)
    animations: Path = field(init=False)
    dashboards: Path = field(init=False)

    # Analysis outputs
    analysis_complexity: Path = field(init=False)
    analysis_comparisons: Path = field(init=False)
    analysis_trends: Path = field(init=False)

    # Performance outputs
    performance_benchmarks: Path = field(init=False)
    performance_profiles: Path = field(init=False)
    performance_anomalies: Path = field(init=False)

    # General outputs
    logs: Path = field(init=False)
    reports: Path = field(init=False)
    data_exports: Path = field(init=False)

    def __post_init__(self):
        """Initialize all output paths based on base directory."""
        self._initialize_paths()

    def _initialize_paths(self):
        """Initialize all output path attributes."""
        # Environment setup
        self.environment_configs = self.base_dir / "environment" / "configs"
        self.environment_reports = self.base_dir / "environment" / "reports"
        self.environment_logs = self.base_dir / "environment" / "logs"

        # Model setup analysis
        self.model_configs = self.base_dir / "models" / "configs"
        self.model_analysis = self.base_dir / "models" / "analysis"
        self.model_performance = self.base_dir / "models" / "performance"

        # Operation
        self.experiments = self.base_dir / "experiments"
        self.runs = self.base_dir / "runs"
        self.experiment_results = self.base_dir / "experiments" / "results"
        self.run_checkpoints = self.base_dir / "runs" / "checkpoints"
        self.run_results = self.base_dir / "runs" / "results"
        self.run_videos = self.base_dir / "runs" / "videos"

        # Visualization
        self.visualizations = self.base_dir / "visualizations"
        self.static_plots = self.base_dir / "visualizations" / "static"
        self.animations = self.base_dir / "visualizations" / "animations"
        self.dashboards = self.base_dir / "visualizations" / "dashboards"

        # Analysis
        self.analysis_complexity = self.base_dir / "analysis" / "complexity"
        self.analysis_comparisons = self.base_dir / "analysis" / "comparisons"
        self.analysis_trends = self.base_dir / "analysis" / "trends"

        # Performance
        self.performance_benchmarks = self.base_dir / "performance" / "benchmarks"
        self.performance_profiles = self.base_dir / "performance" / "profiles"
        self.performance_anomalies = self.base_dir / "performance" / "anomalies"

        # General
        self.logs = self.base_dir / "logs"
        self.reports = self.base_dir / "reports"
        self.data_exports = self.base_dir / "data_exports"

    def ensure_directories_exist(self):
        """Create all output directories if they don't exist."""
        paths_to_create = [
            self.environment_configs, self.environment_reports, self.environment_logs,
            self.model_configs, self.model_analysis, self.model_performance,
            self.experiments, self.runs, self.experiment_results,
            self.run_checkpoints, self.run_results, self.run_videos,
            self.visualizations, self.static_plots, self.animations, self.dashboards,
            self.analysis_complexity, self.analysis_comparisons, self.analysis_trends,
            self.performance_benchmarks, self.performance_profiles, self.performance_anomalies,
            self.logs, self.reports, self.data_exports
        ]

        for path in paths_to_create:
            path.mkdir(parents=True, exist_ok=True)

    def get_path_for_category(self, category: str, subcategory: str = None) -> Path:
        """
        Get the appropriate output path for a given category and subcategory.

        Args:
            category: Main category (e.g., 'environment', 'models', 'experiments')
            subcategory: Optional subcategory within the category

        Returns:
            Path to the appropriate output directory
        """
        category_paths = {
            'environment': {
                'configs': self.environment_configs,
                'reports': self.environment_reports,
                'logs': self.environment_logs,
                None: self.environment_configs
            },
            'models': {
                'configs': self.model_configs,
                'analysis': self.model_analysis,
                'performance': self.model_performance,
                None: self.model_configs
            },
            'experiments': {
                'results': self.experiment_results,
                'reports': self.experiments / "reports",
                'logs': self.experiments / "logs",
                None: self.experiments
            },
            'runs': {
                'checkpoints': self.run_checkpoints,
                'results': self.run_results,
                'videos': self.run_videos,
                None: self.runs
            },
            'visualizations': {
                'static': self.static_plots,
                'animations': self.animations,
                'dashboards': self.dashboards,
                None: self.visualizations
            },
            'analysis': {
                'complexity': self.analysis_complexity,
                'comparisons': self.analysis_comparisons,
                'trends': self.analysis_trends,
                None: self.analysis_complexity
            },
            'performance': {
                'benchmarks': self.performance_benchmarks,
                'profiles': self.performance_profiles,
                'anomalies': self.performance_anomalies,
                None: self.performance_benchmarks
            },
            'logs': {
                None: self.logs
            },
            'reports': {
                None: self.reports
            },
            'data_exports': {
                None: self.data_exports
            }
        }

        if category not in category_paths:
            raise ValueError(f"Unknown category: {category}")

        if subcategory and subcategory not in category_paths[category]:
            # If subcategory doesn't exist, use the category default
            return category_paths[category][None]

        return category_paths[category].get(subcategory, category_paths[category][None])

    def get_log_path(self, module_name: str, filename: str = None) -> Path:
        """
        Get a log file path for a specific module.

        Args:
            module_name: Name of the module (e.g., 'environment_setup', 'model_runner')
            filename: Optional specific filename

        Returns:
            Path to the log file
        """
        if filename is None:
            filename = f"{module_name}.log"

        return self.logs / module_name / filename

    def get_report_path(self, module_name: str, report_name: str, extension: str = "txt") -> Path:
        """
        Get a report file path for a specific module.

        Args:
            module_name: Name of the module
            report_name: Name of the report
            extension: File extension (default: 'txt')

        Returns:
            Path to the report file
        """
        return self.reports / module_name / f"{report_name}.{extension}"

    def set_base_dir(self, new_base_dir: Union[str, Path]):
        """
        Update the base directory and reinitialize all paths.

        Args:
            new_base_dir: New base directory path
        """
        self.base_dir = Path(new_base_dir)
        self._initialize_paths()

    def to_dict(self) -> Dict[str, Any]:
        """Convert paths configuration to dictionary."""
        return {
            'base_dir': str(self.base_dir),
            'environment_configs': str(self.environment_configs),
            'environment_reports': str(self.environment_reports),
            'environment_logs': str(self.environment_logs),
            'model_configs': str(self.model_configs),
            'model_analysis': str(self.model_analysis),
            'model_performance': str(self.model_performance),
            'experiments': str(self.experiments),
            'runs': str(self.runs),
            'experiment_results': str(self.experiment_results),
            'run_checkpoints': str(self.run_checkpoints),
            'run_results': str(self.run_results),
            'run_videos': str(self.run_videos),
            'visualizations': str(self.visualizations),
            'static_plots': str(self.static_plots),
            'animations': str(self.animations),
            'dashboards': str(self.dashboards),
            'analysis_complexity': str(self.analysis_complexity),
            'analysis_comparisons': str(self.analysis_comparisons),
            'analysis_trends': str(self.analysis_trends),
            'performance_benchmarks': str(self.performance_benchmarks),
            'performance_profiles': str(self.performance_profiles),
            'performance_anomalies': str(self.performance_anomalies),
            'logs': str(self.logs),
            'reports': str(self.reports),
            'data_exports': str(self.data_exports)
        }


class OutputConfigManager:
    """
    Manager for output configuration across all AXIOM extension modules.

    This class provides a centralized way to manage output directories and
    ensures all modules use consistent output locations.
    """

    _instance: Optional['OutputConfigManager'] = None
    _paths: OutputPaths = None

    def __new__(cls, base_dir: Optional[Union[str, Path]] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the output configuration manager.

        Args:
            base_dir: Base directory for outputs (optional, uses default if None)
        """
        if hasattr(self, '_initialized') and self._initialized:
            return

        if base_dir is None:
            # Default to extra/output relative to current working directory
            current_dir = Path.cwd()
            if current_dir.name == 'extra':
                base_dir = current_dir / "output"
            elif 'extra' in str(current_dir):
                # Find the extra directory in the path
                parts = current_dir.parts
                extra_index = parts.index('extra')
                base_dir = Path(*parts[:extra_index+1]) / "output"
            else:
                base_dir = Path("extra/output")

        self._paths = OutputPaths(base_dir=base_dir)
        self._paths.ensure_directories_exist()
        self._initialized = True
        logger.info(f"Initialized output configuration with base directory: {base_dir}")

    @property
    def paths(self) -> OutputPaths:
        """Get the output paths configuration."""
        return self._paths

    def set_base_dir(self, new_base_dir: Union[str, Path]):
        """
        Update the base output directory.

        Args:
            new_base_dir: New base directory path
        """
        self._paths.set_base_dir(new_base_dir)
        self._paths.ensure_directories_exist()
        logger.info(f"Updated base output directory to: {new_base_dir}")

    def get_path_for(self, category: str, subcategory: str = None) -> Path:
        """
        Get output path for a specific category and subcategory.

        Args:
            category: Main category
            subcategory: Optional subcategory

        Returns:
            Path to the appropriate output directory
        """
        return self._paths.get_path_for_category(category, subcategory)

    def create_module_logger(self, module_name: str) -> logging.Logger:
        """
        Create a logger configured to output to the appropriate log directory.

        Args:
            module_name: Name of the module

        Returns:
            Configured logger instance
        """
        module_logger = logging.getLogger(module_name)

        # Create log directory for this module
        log_dir = self._paths.logs / module_name
        log_dir.mkdir(parents=True, exist_ok=True)

        # Add file handler if not already present
        if not any(isinstance(h, logging.FileHandler) for h in module_logger.handlers):
            log_file = log_dir / f"{module_name}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            module_logger.addHandler(file_handler)
            module_logger.setLevel(logging.INFO)

        return module_logger


# Global instance for easy access
output_config = OutputConfigManager()
