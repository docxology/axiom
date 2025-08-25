"""
Universal Virtual Environment (UV) Manager for AXIOM Extensions

Provides comprehensive UV methods for setup, environment management,
and advanced analytics integration for AXIOM architecture extensions.
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict, field
import platform
import psutil
import shutil

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class UVConfig:
    """Configuration for Universal Virtual Environment setup."""
    name: str
    python_version: str = "3.11"
    uv_version: str = "latest"
    dependencies: List[str] = field(default_factory=list)
    dev_dependencies: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    gpu_support: bool = False
    memory_requirements_gb: int = 8
    storage_requirements_gb: int = 10
    network_requirements: bool = True
    visualization_support: bool = True
    analytics_support: bool = True
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UVConfig':
        """Create config from dictionary."""
        return cls(**data)


class UVEnvironmentManager:
    """
    Universal Virtual Environment Manager for AXIOM extensions.

    This class provides comprehensive UV methods for:
    - Environment setup and configuration
    - Dependency management with UV
    - System resource monitoring
    - Advanced analytics integration
    - Visualization environment setup
    - Cross-platform compatibility
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize UV Environment Manager.

        Args:
            config_path: Path to environment configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.extra_root = Path(__file__).parent.parent.parent
        self.config = None

        # Setup UV paths
        self._setup_uv_paths()

        # Initialize logging
        self._setup_logging()

        # Load configuration if provided
        if self.config_path and self.config_path.exists():
            self.load_config(self.config_path)

    def _setup_uv_paths(self):
        """Setup UV-related paths and directories."""
        self.uv_cache_dir = Path.home() / ".cache" / "uv"
        self.uv_config_dir = Path.home() / ".config" / "uv"
        self.project_uv_lock = self.project_root / "uv.lock"
        self.project_pyproject = self.project_root / "pyproject.toml"

        # Create necessary directories
        for path in [self.uv_cache_dir, self.uv_config_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        """Setup comprehensive logging for UV operations."""
        # Create logs directory
        logs_dir = self.extra_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Configure logging
        log_file = logs_dir / "uv_environment.log"
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    def load_config(self, config_path: Union[str, Path]) -> UVConfig:
        """
        Load UV environment configuration.

        Args:
            config_path: Path to configuration file

        Returns:
            UVConfig object
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            config_data = json.load(f)

        self.config = UVConfig.from_dict(config_data)
        logger.info(f"Loaded UV configuration: {self.config.name}")
        return self.config

    def create_default_config(self, name: str = "axiom_uv_environment") -> UVConfig:
        """
        Create a default UV configuration for AXIOM.

        Args:
            name: Configuration name

        Returns:
            UVConfig object
        """
        config = UVConfig(
            name=name,
            python_version="3.11",
            dependencies=[
                "jax[cpu]",
                "numpy",
                "matplotlib",
                "networkx",
                "scipy",
                "pandas",
                "seaborn",
                "plotly",
                "dash",
                "jupyter",
                "ipykernel",
                "black",
                "isort",
                "flake8",
                "mypy",
                "pytest",
                "pytest-cov",
                "sphinx",
                "sphinx-rtd-theme",
            ],
            dev_dependencies=[
                "pre-commit",
                "jupyterlab",
                "tensorboard",
                "wandb",
            ],
            environment_variables={
                "AXIOM_EXTRA_ROOT": str(self.extra_root),
                "AXIOM_PROJECT_ROOT": str(self.project_root),
                "JAX_PLATFORM_NAME": "cpu",
                "CUDA_VISIBLE_DEVICES": "",
            },
            gpu_support=False,
            memory_requirements_gb=8,
            storage_requirements_gb=10,
            visualization_support=True,
            analytics_support=True,
            description="Universal Virtual Environment for AXIOM Extensions"
        )

        self.config = config
        logger.info(f"Created default UV configuration: {name}")
        return config

    def check_uv_availability(self) -> Tuple[bool, Optional[str]]:
        """
        Check if UV is available and get version.

        Returns:
            Tuple of (available: bool, version: Optional[str])
        """
        try:
            result = subprocess.run(
                ["uv", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            version = result.stdout.strip()
            logger.info(f"UV available: {version}")
            return True, version
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("UV not found in system PATH")
            return False, None

    def install_uv(self) -> bool:
        """
        Install UV if not available.

        Returns:
            True if installation successful
        """
        logger.info("Installing UV...")

        try:
            # Install UV using curl
            install_cmd = [
                "curl", "-LsSf",
                "https://astral.sh/uv/install.sh"
            ]

            result = subprocess.run(
                install_cmd,
                capture_output=True,
                text=True,
                shell=True
            )

            if result.returncode == 0:
                logger.info("UV installation completed successfully")
                return True
            else:
                logger.error(f"UV installation failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error installing UV: {e}")
            return False

    def create_uv_project(self, project_name: str = "axiom_extensions") -> bool:
        """
        Create a new UV project.

        Args:
            project_name: Name of the project

        Returns:
            True if project creation successful
        """
        if not self.check_uv_availability()[0]:
            logger.error("UV not available. Please install UV first.")
            return False

        project_path = self.extra_root / project_name

        if project_path.exists():
            logger.warning(f"Project already exists: {project_path}")
            return True

        try:
            # Create UV project
            subprocess.run(
                ["uv", "init", str(project_path), "--no-readme"],
                check=True,
                capture_output=True,
                text=True
            )

            logger.info(f"Created UV project: {project_path}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create UV project: {e}")
            return False

    def setup_python_environment(self) -> bool:
        """
        Setup Python environment with UV.

        Returns:
            True if setup successful
        """
        if not self.config:
            logger.error("No configuration loaded. Please load or create a config first.")
            return False

        logger.info(f"Setting up Python environment for: {self.config.name}")

        try:
            # Create virtual environment
            venv_path = self.extra_root / ".venv"

            subprocess.run([
                "uv", "venv", str(venv_path),
                "--python", self.config.python_version
            ], check=True)

            logger.info(f"Created virtual environment: {venv_path}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to setup Python environment: {e}")
            return False

    def install_dependencies(self, dev: bool = False) -> bool:
        """
        Install project dependencies using UV.

        Args:
            dev: Whether to install development dependencies

        Returns:
            True if installation successful
        """
        if not self.config:
            logger.error("No configuration loaded. Please load or create a config first.")
            return False

        logger.info("Installing dependencies with UV...")

        try:
            deps = self.config.dependencies.copy()
            if dev:
                deps.extend(self.config.dev_dependencies)

            # Install dependencies
            for dep in deps:
                logger.info(f"Installing: {dep}")
                subprocess.run([
                    "uv", "add", dep
                ], check=True, cwd=self.extra_root)

            logger.info("All dependencies installed successfully")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False

    def install_from_requirements(self, requirements_path: Union[str, Path]) -> bool:
        """
        Install dependencies from requirements file using UV.

        Args:
            requirements_path: Path to requirements file

        Returns:
            True if installation successful
        """
        requirements_path = Path(requirements_path)
        if not requirements_path.exists():
            logger.error(f"Requirements file not found: {requirements_path}")
            return False

        logger.info(f"Installing from requirements: {requirements_path}")

        try:
            subprocess.run([
                "uv", "pip", "install", "-r", str(requirements_path)
            ], check=True, cwd=self.extra_root)

            logger.info("Requirements installed successfully")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install requirements: {e}")
            return False

    def sync_environment(self) -> bool:
        """
        Sync environment with project dependencies.

        Returns:
            True if sync successful
        """
        logger.info("Syncing environment with UV...")

        try:
            subprocess.run([
                "uv", "sync"
            ], check=True, cwd=self.extra_root)

            logger.info("Environment synced successfully")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to sync environment: {e}")
            return False

    def run_uv_command(self, command: List[str], **kwargs) -> Tuple[bool, str]:
        """
        Run a custom UV command.

        Args:
            command: UV command as list of strings
            **kwargs: Additional subprocess arguments

        Returns:
            Tuple of (success: bool, output: str)
        """
        try:
            result = subprocess.run(
                ["uv"] + command,
                cwd=self.extra_root,
                capture_output=True,
                text=True,
                **kwargs
            )

            if result.returncode == 0:
                logger.info(f"UV command successful: {' '.join(command)}")
                return True, result.stdout
            else:
                logger.error(f"UV command failed: {' '.join(command)}")
                return False, result.stderr

        except Exception as e:
            logger.error(f"Error running UV command: {e}")
            return False, str(e)

    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information.

        Returns:
            Dictionary with system information
        """
        info = {
            "platform": platform.platform(),
            "python_version": sys.version,
            "cpu_count": os.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "disk_free_gb": psutil.disk_usage('/').free / (1024**3),
        }

        # Check GPU availability
        try:
            import torch
            info["gpu_available"] = torch.cuda.is_available()
            info["gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except ImportError:
            info["gpu_available"] = False
            info["gpu_count"] = 0

        # Check UV availability
        uv_available, uv_version = self.check_uv_availability()
        info["uv_available"] = uv_available
        info["uv_version"] = uv_version

        return info

    def validate_environment(self) -> Dict[str, Any]:
        """
        Validate the current environment setup.

        Returns:
            Dictionary with validation results
        """
        if not self.config:
            return {"valid": False, "error": "No configuration loaded"}

        validation = {
            "valid": True,
            "checks": {}
        }

        # Check Python version
        current_version = sys.version_info
        required_version = tuple(map(int, self.config.python_version.split('.')))
        validation["checks"]["python_version"] = {
            "current": f"{current_version.major}.{current_version.minor}.{current_version.micro}",
            "required": self.config.python_version,
            "valid": current_version >= required_version
        }

        # Check memory
        memory_gb = psutil.virtual_memory().total / (1024**3)
        validation["checks"]["memory"] = {
            "current_gb": memory_gb,
            "required_gb": self.config.memory_requirements_gb,
            "valid": memory_gb >= self.config.memory_requirements_gb
        }

        # Check storage
        disk_free_gb = psutil.disk_usage('/').free / (1024**3)
        validation["checks"]["storage"] = {
            "free_gb": disk_free_gb,
            "required_gb": self.config.storage_requirements_gb,
            "valid": disk_free_gb >= self.config.storage_requirements_gb
        }

        # Check UV availability
        uv_available, uv_version = self.check_uv_availability()
        validation["checks"]["uv"] = {
            "available": uv_available,
            "version": uv_version,
            "valid": uv_available
        }

        # Check visualization dependencies
        viz_deps = ["matplotlib", "networkx", "plotly"]
        viz_status = {}
        for dep in viz_deps:
            try:
                __import__(dep)
                viz_status[dep] = True
            except ImportError:
                viz_status[dep] = False

        validation["checks"]["visualization"] = {
            "dependencies": viz_status,
            "valid": all(viz_status.values())
        }

        # Overall validation
        validation["valid"] = all(
            check.get("valid", False)
            for check in validation["checks"].values()
        )

        return validation

    def setup_analytics_integration(self) -> bool:
        """
        Setup advanced analytics integration.

        Returns:
            True if setup successful
        """
        logger.info("Setting up analytics integration...")

        try:
            # Setup analytics directory
            analytics_dir = self.extra_root / "analytics"
            analytics_dir.mkdir(parents=True, exist_ok=True)

            # Create analytics configuration
            analytics_config = {
                "enabled": True,
                "backends": ["wandb", "tensorboard"],
                "metrics": [
                    "loss", "accuracy", "performance",
                    "memory_usage", "computation_time"
                ],
                "visualization_metrics": [
                    "model_complexity", "state_space_evolution",
                    "architecture_relationships", "performance_trends"
                ]
            }

            config_path = analytics_dir / "analytics_config.json"
            with open(config_path, 'w') as f:
                json.dump(analytics_config, f, indent=2)

            logger.info("Analytics integration setup completed")
            return True

        except Exception as e:
            logger.error(f"Failed to setup analytics integration: {e}")
            return False

    def create_uv_script(self, name: str, content: str) -> Path:
        """
        Create a UV-compatible script.

        Args:
            name: Script name (without .py extension)
            content: Script content

        Returns:
            Path to created script
        """
        scripts_dir = self.extra_root / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)

        script_path = scripts_dir / f"{name}.py"

        with open(script_path, 'w') as f:
            f.write(f"""#!/usr/bin/env python3
{content}
""")

        # Make executable
        script_path.chmod(0o755)

        logger.info(f"Created UV script: {script_path}")
        return script_path

    def export_environment_info(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Export comprehensive environment information.

        Args:
            output_path: Optional output path

        Returns:
            Path to exported file
        """
        if output_path is None:
            output_path = self.extra_root / "environment_info.json"
        else:
            output_path = Path(output_path)

        info = {
            "system_info": self.get_system_info(),
            "validation_results": self.validate_environment(),
            "configuration": self.config.to_dict() if self.config else None,
            "timestamp": str(Path('.').absolute())
        }

        with open(output_path, 'w') as f:
            json.dump(info, f, indent=2, default=str)

        logger.info(f"Environment info exported to: {output_path}")
        return output_path

    def create_interactive_dashboard_setup(self) -> bool:
        """
        Setup for interactive dashboard creation.

        Returns:
            True if setup successful
        """
        logger.info("Setting up interactive dashboard environment...")

        try:
            # Create dashboard directory
            dashboard_dir = self.extra_root / "dashboards"
            dashboard_dir.mkdir(parents=True, exist_ok=True)

            # Create dashboard configuration
            dashboard_config = {
                "enabled": True,
                "port": 8050,
                "host": "localhost",
                "components": [
                    "model_visualizer",
                    "performance_monitor",
                    "state_space_explorer",
                    "architecture_browser"
                ],
                "data_sources": [
                    "model_outputs",
                    "performance_metrics",
                    "visualization_data"
                ]
            }

            config_path = dashboard_dir / "dashboard_config.json"
            with open(config_path, 'w') as f:
                json.dump(dashboard_config, f, indent=2)

            logger.info("Interactive dashboard setup completed")
            return True

        except Exception as e:
            logger.error(f"Failed to setup interactive dashboard: {e}")
            return False

