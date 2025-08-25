"""
Environment Manager for AXIOM Extensions

Provides comprehensive environment setup and configuration management
for AXIOM architecture extensions and experiments.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class EnvironmentConfig:
    """Configuration for environment setup."""
    name: str
    python_version: str
    requirements: List[str]
    environment_variables: Dict[str, str]
    gpu_required: bool = False
    memory_gb: int = 8
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnvironmentConfig':
        """Create config from dictionary."""
        return cls(**data)


class EnvironmentManager:
    """
    Manages environment setup and configuration for AXIOM extensions.

    This class provides methods to:
    - Validate environment configurations
    - Set up environments with required dependencies
    - Manage environment variables
    - Check system compatibility
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize environment manager.

        Args:
            config_path: Path to environment configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self.logger = logging.getLogger(__name__)
        self._current_config: Optional[EnvironmentConfig] = None

    def load_config(self, config_path: Union[str, Path]) -> EnvironmentConfig:
        """
        Load environment configuration from file.

        Args:
            config_path: Path to configuration file

        Returns:
            EnvironmentConfig: Loaded configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is invalid JSON
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self._current_config = EnvironmentConfig.from_dict(data)
            self.logger.info(f"Loaded environment config: {self._current_config.name}")
            return self._current_config

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in config file {config_path}: {e}")
            raise

    def save_config(self, config: EnvironmentConfig, config_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Save environment configuration to file.

        Args:
            config: Configuration to save
            config_path: Path to save configuration (if None, uses output config)

        Returns:
            Path to the saved configuration file
        """
        try:
            from ..output_config import output_config
            if config_path is None:
                # Use centralized output configuration
                config_dir = output_config.get_path_for("environment", "configs")
                config_path = config_dir / f"{config.name}.json"
            else:
                config_path = Path(config_path)
        except ImportError:
            # Fallback to default behavior if output_config is not available
            if config_path is None:
                config_path = Path.cwd() / "environment_configs" / f"{config.name}.json"
            else:
                config_path = Path(config_path)

        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved environment config to: {config_path}")
        return config_path

    def validate_environment(self, config: Optional[EnvironmentConfig] = None) -> List[str]:
        """
        Validate current environment against configuration.

        Args:
            config: Configuration to validate against (uses current if None)

        Returns:
            List of validation error messages (empty if valid)
        """
        if config is None:
            if self._current_config is None:
                return ["No configuration loaded"]
            config = self._current_config

        errors = []

        # Check Python version
        current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
        if current_python != config.python_version:
            errors.append(
                f"Python version mismatch: required {config.python_version}, "
                f"current {current_python}"
            )

        # Check GPU requirement
        if config.gpu_required:
            try:
                import torch
                if not torch.cuda.is_available():
                    errors.append("GPU required but CUDA not available")
            except ImportError:
                errors.append("GPU required but PyTorch not installed")

        # Check memory (basic check)
        try:
            import psutil
            available_memory = psutil.virtual_memory().available / (1024**3)  # GB
            if available_memory < config.memory_gb:
                errors.append(
                    f"Insufficient memory: required {config.memory_gb}GB, "
                    f"available {available_memory:.1f}GB"
                )
        except ImportError:
            self.logger.warning("psutil not available for memory check")

        return errors

    def setup_environment_variables(self, config: Optional[EnvironmentConfig] = None) -> None:
        """
        Set up environment variables from configuration.

        Args:
            config: Configuration with environment variables (uses current if None)
        """
        if config is None:
            if self._current_config is None:
                raise ValueError("No configuration loaded")
            config = self._current_config

        for key, value in config.environment_variables.items():
            os.environ[key] = value
            self.logger.info(f"Set environment variable: {key}")

    def get_environment_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about current environment.

        Returns:
            Dictionary with environment information
        """
        info = {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "python_executable": sys.executable,
            "platform": sys.platform,
            "environment_variables": dict(os.environ),
        }

        # Try to get additional system info
        try:
            import platform
            info["system"] = platform.system()
            info["processor"] = platform.processor()
            info["architecture"] = platform.architecture()
        except Exception as e:
            self.logger.warning(f"Could not get system info: {e}")

        # Try to get GPU info
        try:
            import torch
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["cuda_version"] = torch.version.cuda
                info["gpu_count"] = torch.cuda.device_count()
                info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        except ImportError:
            info["cuda_available"] = False
        except Exception as e:
            self.logger.warning(f"Could not get GPU info: {e}")

        # Try to get memory info
        try:
            import psutil
            mem = psutil.virtual_memory()
            info["total_memory_gb"] = mem.total / (1024**3)
            info["available_memory_gb"] = mem.available / (1024**3)
        except ImportError:
            pass
        except Exception as e:
            self.logger.warning(f"Could not get memory info: {e}")

        return info

    def create_default_config(self, name: str = "axiom_extension") -> EnvironmentConfig:
        """
        Create a default environment configuration.

        Args:
            name: Name for the configuration

        Returns:
            EnvironmentConfig: Default configuration
        """
        return EnvironmentConfig(
            name=name,
            python_version="3.11",
            requirements=[
                "numpy>=1.21.0",
                "torch>=2.0.0",
                "matplotlib>=3.5.0",
                "wandb>=0.15.0",
                "psutil>=5.8.0",
                "pytest>=7.0.0",
            ],
            environment_variables={
                "AXIOM_ENV": "development",
                "PYTHONPATH": str(Path.cwd()),
            },
            gpu_required=False,
            memory_gb=8,
            description="Default configuration for AXIOM extensions"
        )
