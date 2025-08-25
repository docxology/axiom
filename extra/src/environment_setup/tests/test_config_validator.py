"""
Tests for Configuration Validator

Comprehensive tests for the ConfigValidator class.
"""

import json
import tempfile
import unittest
from pathlib import Path

# Add the src directory to the path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from environment_setup.config_validator import ConfigValidator, ValidationError


class TestConfigValidator(unittest.TestCase):
    """Tests for ConfigValidator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = ConfigValidator()
        self.valid_config = {
            'name': 'test_env',
            'python_version': '3.11',
            'requirements': ['numpy>=1.21.0', 'torch>=2.0.0'],
            'environment_variables': {'TEST_VAR': 'test_value'},
            'gpu_required': False,
            'memory_gb': 8,
            'description': 'Test configuration'
        }

    def test_validate_valid_config(self):
        """Test validating a completely valid configuration."""
        errors = self.validator.validate_config_dict(self.valid_config)
        self.assertEqual(len(errors), 0)

    def test_validate_missing_required_fields(self):
        """Test validation with missing required fields."""
        invalid_config = {'name': 'test'}  # Missing required fields

        errors = self.validator.validate_config_dict(invalid_config)

        required_fields = ['python_version', 'requirements', 'environment_variables']
        for field in required_fields:
            self.assertTrue(any(field in error for error in errors))

    def test_validate_name_field(self):
        """Test name field validation."""
        # Test empty name
        config = self.valid_config.copy()
        config['name'] = ''
        errors = self.validator.validate_config_dict(config)
        self.assertTrue(any('cannot be empty' in error for error in errors))

        # Test non-string name
        config['name'] = 123
        errors = self.validator.validate_config_dict(config)
        self.assertTrue(any('must be a string' in error for error in errors))

        # Test name too long
        config['name'] = 'a' * 51
        errors = self.validator.validate_config_dict(config)
        self.assertTrue(any('50 characters or less' in error for error in errors))

        # Test invalid characters
        config['name'] = 'test env with spaces!'
        errors = self.validator.validate_config_dict(config)
        self.assertTrue(any('can only contain' in error for error in errors))

    def test_validate_python_version_field(self):
        """Test Python version field validation."""
        config = self.valid_config.copy()

        # Test valid versions
        valid_versions = ['3.9', '3.11', '3.10.5']
        for version in valid_versions:
            config['python_version'] = version
            errors = self.validator.validate_config_dict(config)
            self.assertEqual(len(errors), 0, f"Version {version} should be valid")

        # Test invalid versions
        invalid_versions = ['2.7', '3', '3.11.1.1', 'python3.11', '']
        for version in invalid_versions:
            config['python_version'] = version
            errors = self.validator.validate_config_dict(config)
            self.assertTrue(len(errors) > 0, f"Version {version} should be invalid")

        # Test non-string version
        config['python_version'] = 3.11
        errors = self.validator.validate_config_dict(config)
        self.assertTrue(any('must be a string' in error for error in errors))

    def test_validate_requirements_field(self):
        """Test requirements field validation."""
        config = self.valid_config.copy()

        # Test non-list requirements
        config['requirements'] = 'numpy>=1.21.0'
        errors = self.validator.validate_config_dict(config)
        self.assertTrue(any('must be a list' in error for error in errors))

        # Test empty requirement
        config['requirements'] = ['']
        errors = self.validator.validate_config_dict(config)
        self.assertTrue(any('cannot be empty' in error for error in errors))

        # Test non-string requirement
        config['requirements'] = [123]
        errors = self.validator.validate_config_dict(config)
        self.assertTrue(any('must be a string' in error for error in errors))

        # Test valid requirements
        valid_requirements = [
            'numpy',
            'numpy>=1.21.0',
            'torch>=2.0.0',
            'matplotlib>=3.5.0,<4.0.0',
            'package[submodule]>=1.0',
            'numpy>=1.21.0; python_version >= "3.8"'
        ]

        for req in valid_requirements:
            config['requirements'] = [req]
            errors = self.validator.validate_config_dict(config)
            self.assertEqual(len(errors), 0, f"Requirement '{req}' should be valid")

    def test_validate_environment_variables_field(self):
        """Test environment variables field validation."""
        config = self.valid_config.copy()

        # Test non-dict environment variables
        config['environment_variables'] = ['TEST_VAR=test']
        errors = self.validator.validate_config_dict(config)
        self.assertTrue(any('must be a dictionary' in error for error in errors))

        # Test non-string key
        config['environment_variables'] = {123: 'value'}
        errors = self.validator.validate_config_dict(config)
        self.assertTrue(any('key must be string' in error for error in errors))

        # Test non-string value
        config['environment_variables'] = {'TEST_VAR': 123}
        errors = self.validator.validate_config_dict(config)
        self.assertTrue(any('value must be string' in error for error in errors))

        # Test empty key
        config['environment_variables'] = {'': 'value'}
        errors = self.validator.validate_config_dict(config)
        self.assertTrue(any('key cannot be empty' in error for error in errors))

        # Test invalid environment variable names
        invalid_names = ['test', 'test-var', 'test.var', 'Test']
        for name in invalid_names:
            config['environment_variables'] = {name: 'value'}
            errors = self.validator.validate_config_dict(config)
            self.assertTrue(len(errors) > 0, f"Environment variable name '{name}' should be invalid")

        # Test valid environment variable names
        valid_names = ['TEST_VAR', 'TEST123', 'MY_VAR', 'API_KEY_1']
        for name in valid_names:
            config['environment_variables'] = {name: 'value'}
            errors = self.validator.validate_config_dict(config)
            self.assertEqual(len(errors), 0, f"Environment variable name '{name}' should be valid")

    def test_validate_optional_fields(self):
        """Test validation of optional fields."""
        config = self.valid_config.copy()

        # Test gpu_required field
        config['gpu_required'] = 'yes'  # Should be boolean
        errors = self.validator.validate_config_dict(config)
        self.assertTrue(any('gpu_required must be a boolean' in error for error in errors))

        # Test memory_gb field
        config['gpu_required'] = False
        config['memory_gb'] = '8'  # Should be int
        errors = self.validator.validate_config_dict(config)
        self.assertTrue(any('memory_gb must be a positive integer' in error for error in errors))

        config['memory_gb'] = -1  # Should be positive
        errors = self.validator.validate_config_dict(config)
        self.assertTrue(any('memory_gb must be a positive integer' in error for error in errors))

        # Test description field
        config['memory_gb'] = 8
        config['description'] = 123  # Should be string
        errors = self.validator.validate_config_dict(config)
        self.assertTrue(any('description must be a string' in error for error in errors))

        config['description'] = 'a' * 501  # Too long
        errors = self.validator.validate_config_dict(config)
        self.assertTrue(any('description must be 500 characters or less' in error for error in errors))

    def test_validate_config_file(self):
        """Test validating configuration files."""
        # Test non-existent file
        errors = self.validator.validate_config_file("/nonexistent/path.json")
        self.assertTrue(len(errors) > 0)
        self.assertTrue(any('does not exist' in error for error in errors))

        # Test valid config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.valid_config, f)
            temp_path = Path(f.name)

        try:
            errors = self.validator.validate_config_file(temp_path)
            self.assertEqual(len(errors), 0)
        finally:
            temp_path.unlink()

        # Test invalid JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_path = Path(f.name)

        try:
            errors = self.validator.validate_config_file(temp_path)
            self.assertTrue(len(errors) > 0)
            self.assertTrue(any('Invalid JSON' in error for error in errors))
        finally:
            temp_path.unlink()

    def test_validate_requirements_file(self):
        """Test validating requirements files."""
        # Test non-existent file
        errors = self.validator.validate_requirements_file("/nonexistent/requirements.txt")
        self.assertTrue(len(errors) > 0)
        self.assertTrue(any('does not exist' in error for error in errors))

        # Test valid requirements file
        valid_requirements = [
            'numpy>=1.21.0',
            'torch>=2.0.0',
            'matplotlib>=3.5.0',
            '# This is a comment',
            '',
            'pandas'
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for req in valid_requirements:
                f.write(f"{req}\n")
            temp_path = Path(f.name)

        try:
            errors = self.validator.validate_requirements_file(temp_path)
            self.assertEqual(len(errors), 0)
        finally:
            temp_path.unlink()

        # Test invalid requirements file
        invalid_requirements = [
            'invalid requirement format !!!',
            'another invalid format @#$%'
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for req in invalid_requirements:
                f.write(f"{req}\n")
            temp_path = Path(f.name)

        try:
            errors = self.validator.validate_requirements_file(temp_path)
            self.assertEqual(len(errors), len(invalid_requirements))
        finally:
            temp_path.unlink()

    def test_is_valid_config(self):
        """Test is_valid_config method."""
        # Test valid config
        self.assertTrue(self.validator.is_valid_config(self.valid_config))

        # Test invalid config
        invalid_config = {'name': 'test'}  # Missing required fields
        self.assertFalse(self.validator.is_valid_config(invalid_config))

    def test_is_valid_config_file(self):
        """Test is_valid_config_file method."""
        # Test valid config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.valid_config, f)
            temp_path = Path(f.name)

        try:
            self.assertTrue(self.validator.is_valid_config_file(temp_path))
        finally:
            temp_path.unlink()

        # Test invalid config file
        self.assertFalse(self.validator.is_valid_config_file("/nonexistent/path.json"))


if __name__ == '__main__':
    unittest.main()
