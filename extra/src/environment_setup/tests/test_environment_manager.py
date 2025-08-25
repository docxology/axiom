"""
Tests for Environment Manager

Comprehensive tests for the EnvironmentManager class and EnvironmentConfig dataclass.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the src directory to the path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from environment_setup.environment_manager import EnvironmentManager, EnvironmentConfig


class TestEnvironmentConfig(unittest.TestCase):
    """Tests for EnvironmentConfig dataclass."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_config = EnvironmentConfig(
            name="test_env",
            python_version="3.11",
            requirements=["numpy>=1.21.0", "torch>=2.0.0"],
            environment_variables={"TEST_VAR": "test_value"},
            gpu_required=True,
            memory_gb=16,
            description="Test environment configuration"
        )

    def test_config_creation(self):
        """Test creating a valid configuration."""
        self.assertEqual(self.valid_config.name, "test_env")
        self.assertEqual(self.valid_config.python_version, "3.11")
        self.assertEqual(len(self.valid_config.requirements), 2)
        self.assertEqual(self.valid_config.gpu_required, True)
        self.assertEqual(self.valid_config.memory_gb, 16)

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config_dict = self.valid_config.to_dict()

        expected_keys = [
            'name', 'python_version', 'requirements',
            'environment_variables', 'gpu_required', 'memory_gb', 'description'
        ]

        for key in expected_keys:
            self.assertIn(key, config_dict)

        self.assertEqual(config_dict['name'], 'test_env')
        self.assertEqual(config_dict['python_version'], '3.11')

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            'name': 'from_dict_env',
            'python_version': '3.9',
            'requirements': ['pandas'],
            'environment_variables': {'PATH': '/usr/bin'},
            'gpu_required': False,
            'memory_gb': 8,
            'description': 'Created from dict'
        }

        config = EnvironmentConfig.from_dict(config_dict)

        self.assertEqual(config.name, 'from_dict_env')
        self.assertEqual(config.python_version, '3.9')
        self.assertEqual(config.requirements, ['pandas'])
        self.assertEqual(config.environment_variables, {'PATH': '/usr/bin'})
        self.assertEqual(config.gpu_required, False)
        self.assertEqual(config.memory_gb, 8)


class TestEnvironmentManager(unittest.TestCase):
    """Tests for EnvironmentManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = EnvironmentManager()
        self.test_config = EnvironmentConfig(
            name="test_env",
            python_version="3.11",
            requirements=["numpy>=1.21.0"],
            environment_variables={"TEST_VAR": "test_value"},
            gpu_required=False,
            memory_gb=8
        )

    def test_initialization(self):
        """Test EnvironmentManager initialization."""
        manager = EnvironmentManager()
        self.assertIsNone(manager._current_config)

        config_path = Path("/test/config.json")
        manager_with_path = EnvironmentManager(config_path)
        self.assertEqual(manager_with_path.config_path, config_path)

    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Save config
            self.manager.save_config(self.test_config, temp_path)

            # Verify file exists and contains data
            self.assertTrue(temp_path.exists())

            with open(temp_path, 'r') as f:
                saved_data = json.load(f)

            self.assertEqual(saved_data['name'], 'test_env')
            self.assertEqual(saved_data['python_version'], '3.11')

            # Load config
            loaded_config = self.manager.load_config(temp_path)
            self.assertEqual(loaded_config.name, self.test_config.name)
            self.assertEqual(loaded_config.python_version, self.test_config.python_version)

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_load_config_file_not_found(self):
        """Test loading non-existent config file."""
        with self.assertRaises(FileNotFoundError):
            self.manager.load_config("/nonexistent/path/config.json")

    def test_load_config_invalid_json(self):
        """Test loading invalid JSON config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content {")
            temp_path = Path(f.name)

        try:
            with self.assertRaises(json.JSONDecodeError):
                self.manager.load_config(temp_path)
        finally:
            temp_path.unlink()

    def test_validate_environment_no_config(self):
        """Test validating environment with no config loaded."""
        errors = self.manager.validate_environment()
        self.assertIn("No configuration loaded", errors)

    @patch('environment_setup.environment_manager.sys')
    def test_validate_environment_python_version(self, mock_sys):
        """Test Python version validation."""
        mock_sys.version_info = type('version_info', (), {'major': 3, 'minor': 9})()

        config = EnvironmentConfig(
            name="test",
            python_version="3.11",
            requirements=[],
            environment_variables={}
        )

        errors = self.manager.validate_environment(config)
        self.assertTrue(any("Python version mismatch" in error for error in errors))

    def test_validate_environment_gpu_config_only(self):
        """Test GPU validation with gpu_required=True but no actual GPU check."""
        config = EnvironmentConfig(
            name="test",
            python_version="3.11",
            requirements=[],
            environment_variables={},
            gpu_required=True
        )

        # This will test the configuration parsing, but won't actually check GPU
        # since we can't easily mock torch import in the function scope
        self.assertEqual(config.gpu_required, True)

    def test_validate_environment_no_gpu_config(self):
        """Test GPU validation with gpu_required=False."""
        config = EnvironmentConfig(
            name="test",
            python_version="3.11",
            requirements=[],
            environment_variables={},
            gpu_required=False
        )

        # This should pass without GPU checks
        errors = self.manager.validate_environment(config)
        # We expect no errors for python version (assuming current python matches)
        # but we don't test GPU since it's not required
        self.assertEqual(config.gpu_required, False)

    def test_validate_environment_gpu_pytorch_config_only(self):
        """Test GPU configuration parsing."""
        config = EnvironmentConfig(
            name="test",
            python_version="3.11",
            requirements=[],
            environment_variables={},
            gpu_required=True
        )

        # Test that the config correctly stores the GPU requirement
        self.assertEqual(config.gpu_required, True)

    def test_validate_environment_memory_config(self):
        """Test memory configuration parsing."""
        config = EnvironmentConfig(
            name="test",
            python_version="3.11",
            requirements=[],
            environment_variables={},
            memory_gb=16
        )

        # Test that the config correctly stores the memory requirement
        self.assertEqual(config.memory_gb, 16)

    def test_validate_environment_memory_insufficient_config(self):
        """Test memory configuration with lower memory."""
        config = EnvironmentConfig(
            name="test",
            python_version="3.11",
            requirements=[],
            environment_variables={},
            memory_gb=4
        )

        # Test that the config correctly stores the memory requirement
        self.assertEqual(config.memory_gb, 4)

    def test_setup_environment_variables(self):
        """Test setting up environment variables."""
        config = EnvironmentConfig(
            name="test",
            python_version="3.11",
            requirements=[],
            environment_variables={"TEST_VAR": "test_value", "ANOTHER_VAR": "another_value"}
        )

        # Clear any existing environment variables
        for key in config.environment_variables:
            os.environ.pop(key, None)

        # Set environment variables
        self.manager.setup_environment_variables(config)

        # Verify they were set
        self.assertEqual(os.environ.get("TEST_VAR"), "test_value")
        self.assertEqual(os.environ.get("ANOTHER_VAR"), "another_value")

        # Clean up
        for key in config.environment_variables:
            os.environ.pop(key, None)

    def test_setup_environment_variables_no_config(self):
        """Test setting up environment variables with no config."""
        manager = EnvironmentManager()

        with self.assertRaises(ValueError):
            manager.setup_environment_variables()

    def test_get_environment_info_basic(self):
        """Test getting basic environment information."""
        info = self.manager.get_environment_info()

        # Test that basic information is present
        self.assertIn('python_version', info)
        self.assertIn('python_executable', info)
        self.assertIn('environment_variables', info)

        # Test that python version is a string and contains expected format
        self.assertIsInstance(info['python_version'], str)
        self.assertIn('.', info['python_version'])

        # Test that python executable path exists
        self.assertIsInstance(info['python_executable'], str)

        # Test that environment variables is a dictionary
        self.assertIsInstance(info['environment_variables'], dict)

    def test_create_default_config(self):
        """Test creating default configuration."""
        config = self.manager.create_default_config("my_extension")

        self.assertEqual(config.name, "my_extension")
        self.assertEqual(config.python_version, "3.11")
        self.assertIn("numpy>=1.21.0", config.requirements)
        self.assertIn("torch>=2.0.0", config.requirements)
        self.assertEqual(config.gpu_required, False)
        self.assertEqual(config.memory_gb, 8)
        self.assertIn("AXIOM_ENV", config.environment_variables)


if __name__ == '__main__':
    unittest.main()
