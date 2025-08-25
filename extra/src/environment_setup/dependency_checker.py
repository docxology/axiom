"""
Dependency Checker for AXIOM Extensions

Provides functionality to check, install, and manage dependencies
for AXIOM architecture extensions.
"""

import subprocess
import sys
import logging
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import importlib.util

logger = logging.getLogger(__name__)


class DependencyError(Exception):
    """Exception raised for dependency-related errors."""
    pass


class DependencyChecker:
    """
    Checks and manages Python package dependencies.

    This class provides methods to:
    - Check if packages are installed
    - Install missing dependencies
    - Verify package versions
    - Manage virtual environments
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize dependency checker.

        Args:
            verbose: Whether to enable verbose logging
        """
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

    def is_package_installed(self, package_name: str) -> bool:
        """
        Check if a Python package is installed.

        Args:
            package_name: Name of the package to check

        Returns:
            True if package is installed, False otherwise
        """
        try:
            importlib.import_module(package_name.replace('-', '_'))
            return True
        except ImportError:
            return False

    def get_package_version(self, package_name: str) -> Optional[str]:
        """
        Get the installed version of a package.

        Args:
            package_name: Name of the package

        Returns:
            Version string if installed, None otherwise
        """
        try:
            module = importlib.import_module(package_name.replace('-', '_'))
            return getattr(module, '__version__', None)
        except ImportError:
            return None

    def check_requirements(self, requirements: List[str]) -> Tuple[List[str], List[str]]:
        """
        Check which requirements are installed and which are missing.

        Args:
            requirements: List of requirement strings

        Returns:
            Tuple of (installed_packages, missing_packages)
        """
        installed = []
        missing = []

        for req in requirements:
            # Extract package name (handle version specifications)
            package_name = req.split('[')[0].split('<')[0].split('>')[0].split('=')[0].split('!')[0].split('~')[0].strip()

            if self.is_package_installed(package_name):
                installed.append(req)
                version = self.get_package_version(package_name)
                if self.verbose:
                    self.logger.info(f"✓ {package_name} is installed" + (f" (version: {version})" if version else ""))
            else:
                missing.append(req)
                if self.verbose:
                    self.logger.warning(f"✗ {package_name} is missing")

        return installed, missing

    def install_package(self, package_spec: str, upgrade: bool = False) -> bool:
        """
        Install a single package.

        Args:
            package_spec: Package specification (e.g., "numpy>=1.21.0")
            upgrade: Whether to upgrade if already installed

        Returns:
            True if installation successful, False otherwise
        """
        cmd = [sys.executable, '-m', 'pip', 'install']

        if upgrade:
            cmd.append('--upgrade')

        cmd.append(package_spec)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=300  # 5 minute timeout
            )

            if self.verbose:
                self.logger.info(f"Successfully installed: {package_spec}")

            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install {package_spec}: {e}")
            if e.stderr:
                self.logger.error(f"Error output: {e.stderr}")
            return False
        except subprocess.TimeoutExpired:
            self.logger.error(f"Timeout installing {package_spec}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error installing {package_spec}: {e}")
            return False

    def install_packages(self, packages: List[str], upgrade: bool = False) -> Tuple[List[str], List[str]]:
        """
        Install multiple packages.

        Args:
            packages: List of package specifications
            upgrade: Whether to upgrade if already installed

        Returns:
            Tuple of (successfully_installed, failed_packages)
        """
        successful = []
        failed = []

        for package in packages:
            if self.install_package(package, upgrade):
                successful.append(package)
            else:
                failed.append(package)

        return successful, failed

    def install_requirements_file(self, requirements_path: Union[str, Path], upgrade: bool = False) -> bool:
        """
        Install packages from a requirements.txt file.

        Args:
            requirements_path: Path to requirements file
            upgrade: Whether to upgrade existing packages

        Returns:
            True if all packages installed successfully, False otherwise
        """
        requirements_path = Path(requirements_path)

        if not requirements_path.exists():
            raise DependencyError(f"Requirements file not found: {requirements_path}")

        cmd = [sys.executable, '-m', 'pip', 'install']

        if upgrade:
            cmd.append('--upgrade')

        cmd.extend(['-r', str(requirements_path)])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=600  # 10 minute timeout
            )

            if self.verbose:
                self.logger.info(f"Successfully installed requirements from: {requirements_path}")

            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install requirements from {requirements_path}: {e}")
            if e.stderr:
                self.logger.error(f"Error output: {e.stderr}")
            return False
        except subprocess.TimeoutExpired:
            self.logger.error(f"Timeout installing requirements from {requirements_path}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error installing requirements: {e}")
            return False

    def create_requirements_file(self, packages: List[str], output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Create a requirements.txt file from a list of packages.

        Args:
            packages: List of package specifications
            output_path: Path to write requirements file (if None, uses output config)

        Returns:
            Path to the created requirements file
        """
        try:
            from ..output_config import output_config
            if output_path is None:
                # Use centralized output configuration
                requirements_dir = output_config.get_path_for("environment", "configs")
                output_path = requirements_dir / "requirements.txt"
            else:
                output_path = Path(output_path)
        except ImportError:
            # Fallback to default behavior if output_config is not available
            if output_path is None:
                output_path = Path.cwd() / "requirements.txt"
            else:
                output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for package in packages:
                f.write(f"{package}\n")

        if self.verbose:
            self.logger.info(f"Created requirements file: {output_path}")

        return output_path

    def verify_python_version(self, required_version: str) -> bool:
        """
        Verify that the current Python version meets requirements.

        Args:
            required_version: Required Python version (e.g., "3.11")

        Returns:
            True if version requirement is met, False otherwise
        """
        current_version = f"{sys.version_info.major}.{sys.version_info.minor}"

        try:
            current_parts = [int(x) for x in current_version.split('.')]
            required_parts = [int(x) for x in required_version.split('.')]

            # Pad shorter version with zeros
            while len(current_parts) < len(required_parts):
                current_parts.append(0)
            while len(required_parts) < len(current_parts):
                required_parts.append(0)

            return current_parts >= required_parts

        except ValueError:
            self.logger.error(f"Invalid version format: {required_version}")
            return False

    def check_system_dependencies(self) -> Dict[str, bool]:
        """
        Check for common system-level dependencies.

        Returns:
            Dictionary mapping dependency names to availability status
        """
        system_deps = {
            'git': self._check_command_available('git'),
            'curl': self._check_command_available('curl'),
            'wget': self._check_command_available('wget'),
            'make': self._check_command_available('make'),
            'gcc': self._check_command_available('gcc'),
        }

        return system_deps

    def _check_command_available(self, command: str) -> bool:
        """
        Check if a system command is available.

        Args:
            command: Command name to check

        Returns:
            True if command is available, False otherwise
        """
        try:
            subprocess.run(
                [command, '--version'],
                capture_output=True,
                check=True,
                timeout=10
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def get_environment_summary(self) -> Dict[str, any]:
        """
        Get a summary of the current environment and dependencies.

        Returns:
            Dictionary with environment information
        """
        summary = {
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'python_executable': sys.executable,
            'system_dependencies': self.check_system_dependencies(),
        }

        # Check for common ML/AI packages
        ml_packages = [
            'torch', 'torchvision', 'torchaudio',
            'numpy', 'scipy', 'pandas',
            'matplotlib', 'seaborn',
            'scikit-learn', 'wandb',
            'pytest', 'black', 'flake8'
        ]

        package_status = {}
        for package in ml_packages:
            installed = self.is_package_installed(package)
            version = self.get_package_version(package) if installed else None
            package_status[package] = {
                'installed': installed,
                'version': version
            }

        summary['python_packages'] = package_status

        return summary
