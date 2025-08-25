"""
Configuration Validator for AXIOM Extensions

Provides validation functionality for environment configurations
and system requirements.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Exception raised for configuration validation errors."""
    pass


class ConfigValidator:
    """
    Validates environment configurations and system requirements.

    This class provides methods to:
    - Validate configuration file formats
    - Check requirement specifications
    - Validate environment variable formats
    - Perform comprehensive configuration validation
    """

    # Valid Python version pattern (e.g., "3.11", "3.9.7")
    PYTHON_VERSION_PATTERN = re.compile(r'^3\.\d+(\.\d+)?$')

    # Valid package requirement pattern (simplified)
    REQUIREMENT_PATTERN = re.compile(r'^[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?(\[[^\]]+\])?([<>=!~]+[^,]+)?(,[<>=!~]+[^,]+)*$')

    def __init__(self):
        """Initialize the configuration validator."""
        self.logger = logging.getLogger(__name__)

    def validate_config_dict(self, config_dict: Dict[str, Any]) -> List[str]:
        """
        Validate a configuration dictionary.

        Args:
            config_dict: Configuration dictionary to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Required fields
        required_fields = ['name', 'python_version', 'requirements', 'environment_variables']
        for field in required_fields:
            if field not in config_dict:
                errors.append(f"Missing required field: {field}")

        if errors:
            return errors

        # Validate name
        errors.extend(self._validate_name(config_dict.get('name', '')))

        # Validate Python version
        errors.extend(self._validate_python_version(config_dict.get('python_version', '')))

        # Validate requirements
        errors.extend(self._validate_requirements(config_dict.get('requirements', [])))

        # Validate environment variables
        errors.extend(self._validate_environment_variables(config_dict.get('environment_variables', {})))

        # Validate optional fields
        errors.extend(self._validate_optional_fields(config_dict))

        return errors

    def _validate_name(self, name: str) -> List[str]:
        """Validate configuration name."""
        errors = []

        if not isinstance(name, str):
            errors.append("Name must be a string")
            return errors

        if not name.strip():
            errors.append("Name cannot be empty")
            return errors

        if len(name) > 50:
            errors.append("Name must be 50 characters or less")

        # Valid characters: alphanumeric, underscore, dash
        if not re.match(r'^[a-zA-Z0-9_-]+$', name):
            errors.append("Name can only contain alphanumeric characters, underscores, and dashes")

        return errors

    def _validate_python_version(self, version: str) -> List[str]:
        """Validate Python version string."""
        errors = []

        if not isinstance(version, str):
            errors.append("Python version must be a string")
            return errors

        if not self.PYTHON_VERSION_PATTERN.match(version):
            errors.append(f"Invalid Python version format: {version} (expected format: 3.X or 3.X.Y)")

        return errors

    def _validate_requirements(self, requirements: List[str]) -> List[str]:
        """Validate requirements list."""
        errors = []

        if not isinstance(requirements, list):
            errors.append("Requirements must be a list")
            return errors

        for i, req in enumerate(requirements):
            if not isinstance(req, str):
                errors.append(f"Requirement at index {i} must be a string")
                continue

            if not req.strip():
                errors.append(f"Requirement at index {i} cannot be empty")
                continue

            # Basic validation - more comprehensive validation would require parsing
            if not self.REQUIREMENT_PATTERN.match(req):
                errors.append(f"Invalid requirement format at index {i}: {req}")

        return errors

    def _validate_environment_variables(self, env_vars: Dict[str, str]) -> List[str]:
        """Validate environment variables dictionary."""
        errors = []

        if not isinstance(env_vars, dict):
            errors.append("Environment variables must be a dictionary")
            return errors

        for key, value in env_vars.items():
            if not isinstance(key, str):
                errors.append(f"Environment variable key must be string: {key}")
                continue

            if not isinstance(value, str):
                errors.append(f"Environment variable value must be string for key: {key}")
                continue

            if not key:
                errors.append("Environment variable key cannot be empty")
                continue

            # Valid environment variable name (POSIX standard)
            if not re.match(r'^[A-Z][A-Z0-9_]*$', key):
                errors.append(f"Invalid environment variable name: {key}")

        return errors

    def _validate_optional_fields(self, config_dict: Dict[str, Any]) -> List[str]:
        """Validate optional configuration fields."""
        errors = []

        # Validate gpu_required
        if 'gpu_required' in config_dict:
            gpu_req = config_dict['gpu_required']
            if not isinstance(gpu_req, bool):
                errors.append("gpu_required must be a boolean")

        # Validate memory_gb
        if 'memory_gb' in config_dict:
            memory = config_dict['memory_gb']
            if not isinstance(memory, int) or memory <= 0:
                errors.append("memory_gb must be a positive integer")

        # Validate description
        if 'description' in config_dict:
            desc = config_dict['description']
            if not isinstance(desc, str):
                errors.append("description must be a string")
            elif len(desc) > 500:
                errors.append("description must be 500 characters or less")

        return errors

    def validate_config_file(self, config_path: Union[str, Path]) -> List[str]:
        """
        Validate a configuration file.

        Args:
            config_path: Path to configuration file

        Returns:
            List of validation error messages (empty if valid)
        """
        config_path = Path(config_path)

        if not config_path.exists():
            return [f"Configuration file does not exist: {config_path}"]

        if not config_path.is_file():
            return [f"Configuration path is not a file: {config_path}"]

        try:
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        except json.JSONDecodeError as e:
            return [f"Invalid JSON in configuration file: {e}"]
        except Exception as e:
            return [f"Error reading configuration file: {e}"]

        return self.validate_config_dict(config_dict)

    def validate_requirements_file(self, requirements_path: Union[str, Path]) -> List[str]:
        """
        Validate a requirements.txt file.

        Args:
            requirements_path: Path to requirements file

        Returns:
            List of validation error messages (empty if valid)
        """
        requirements_path = Path(requirements_path)

        if not requirements_path.exists():
            return [f"Requirements file does not exist: {requirements_path}"]

        try:
            with open(requirements_path, 'r', encoding='utf-8') as f:
                requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        except Exception as e:
            return [f"Error reading requirements file: {e}"]

        errors = []
        for i, req in enumerate(requirements):
            if not self.REQUIREMENT_PATTERN.match(req):
                errors.append(f"Invalid requirement format at line {i+1}: {req}")

        return errors

    def is_valid_config(self, config_dict: Dict[str, Any]) -> bool:
        """
        Check if a configuration dictionary is valid.

        Args:
            config_dict: Configuration dictionary to check

        Returns:
            True if configuration is valid, False otherwise
        """
        return len(self.validate_config_dict(config_dict)) == 0

    def is_valid_config_file(self, config_path: Union[str, Path]) -> bool:
        """
        Check if a configuration file is valid.

        Args:
            config_path: Path to configuration file

        Returns:
            True if configuration file is valid, False otherwise
        """
        return len(self.validate_config_file(config_path)) == 0
