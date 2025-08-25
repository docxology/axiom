"""
Tests for UV Environment Manager

Comprehensive test suite for the Universal Virtual Environment Manager
including setup, configuration, dependency management, and validation.
"""

import unittest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import tempfile
import shutil
import json
import os
import sys

# Add src to path for imports
extra_dir = Path(__file__).parent.parent.parent.parent
src_dir = extra_dir / "src"
sys.path.insert(0, str(src_dir))

from environment_setup.uv_environment_manager import UVEnvironmentManager, UVConfig


class TestUVEnvironmentManager(unittest.TestCase):
    """Test cases for UV Environment Manager."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "test_config.json"
        self.uv_manager = UVEnvironmentManager()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test UV Environment Manager initialization."""
        self.assertIsInstance(self.uv_manager, UVEnvironmentManager)
        self.assertIsNone(self.uv_manager.config)
        self.assertEqual(self.uv_manager.project_root.name, "axiom")
        self.assertEqual(self.uv_manager.extra_root.name, "extra")

    def test_create_default_config(self):
        """Test creating default UV configuration."""
        config = self.uv_manager.create_default_config("test_config")

        self.assertIsInstance(config, UVConfig)
        self.assertEqual(config.name, "test_config")
        self.assertEqual(config.python_version, "3.11")
        self.assertIn("jax[cpu]", config.dependencies)
        self.assertIn("matplotlib", config.dependencies)
        self.assertIn("networkx", config.dependencies)
        self.assertTrue(config.visualization_support)
        self.assertTrue(config.analytics_support)

    def test_load_config(self):
        """Test loading configuration from file."""
        config_data = {
            "name": "test_config",
            "python_version": "3.10",
            "dependencies": ["numpy", "matplotlib"],
            "visualization_support": True
        }

        with open(self.config_path, 'w') as f:
            json.dump(config_data, f)

        config = self.uv_manager.load_config(self.config_path)

        self.assertIsInstance(config, UVConfig)
        self.assertEqual(config.name, "test_config")
        self.assertEqual(config.python_version, "3.10")
        self.assertEqual(config.dependencies, ["numpy", "matplotlib"])

    def test_load_config_file_not_found(self):
        """Test loading configuration from non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.uv_manager.load_config(Path("non_existent.json"))

    @patch('subprocess.run')
    def test_check_uv_availability_available(self, mock_run):
        """Test checking UV availability when available."""
        mock_run.return_value = MagicMock(stdout="uv 0.8.5", returncode=0)

        available, version = self.uv_manager.check_uv_availability()

        self.assertTrue(available)
        self.assertEqual(version, "uv 0.8.5")
        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_check_uv_availability_not_available(self, mock_run):
        """Test checking UV availability when not available."""
        mock_run.side_effect = FileNotFoundError()

        available, version = self.uv_manager.check_uv_availability()

        self.assertFalse(available)
        self.assertIsNone(version)

    @patch('subprocess.run')
    def test_install_uv_success(self, mock_run):
        """Test successful UV installation."""
        mock_run.return_value = MagicMock(returncode=0)

        result = self.uv_manager.install_uv()

        self.assertTrue(result)
        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_install_uv_failure(self, mock_run):
        """Test failed UV installation."""
        mock_run.return_value = MagicMock(returncode=1, stderr="Installation failed")

        result = self.uv_manager.install_uv()

        self.assertFalse(result)

    @patch('subprocess.run')
    def test_create_uv_project_success(self, mock_run):
        """Test successful UV project creation."""
        mock_run.return_value = MagicMock(returncode=0)

        with patch.object(self.uv_manager, 'check_uv_availability', return_value=(True, "uv 0.8.5")):
            result = self.uv_manager.create_uv_project("test_project")

        self.assertTrue(result)

    @patch('subprocess.run')
    def test_create_uv_project_uv_not_available(self, mock_run):
        """Test UV project creation when UV is not available."""
        with patch.object(self.uv_manager, 'check_uv_availability', return_value=(False, None)):
            result = self.uv_manager.create_uv_project("test_project")

        self.assertFalse(result)

    @patch('subprocess.run')
    def test_setup_python_environment_success(self, mock_run):
        """Test successful Python environment setup."""
        self.uv_manager.config = self.uv_manager.create_default_config("test")
        mock_run.return_value = MagicMock(returncode=0)

        result = self.uv_manager.setup_python_environment()

        self.assertTrue(result)
        mock_run.assert_called_once()

    def test_setup_python_environment_no_config(self):
        """Test Python environment setup without configuration."""
        self.uv_manager.config = None

        result = self.uv_manager.setup_python_environment()

        self.assertFalse(result)

    @patch('subprocess.run')
    def test_install_dependencies_success(self, mock_run):
        """Test successful dependency installation."""
        self.uv_manager.config = self.uv_manager.create_default_config("test")
        mock_run.return_value = MagicMock(returncode=0)

        result = self.uv_manager.install_dependencies()

        self.assertTrue(result)
        # Should call subprocess for each dependency
        self.assertEqual(mock_run.call_count, len(self.uv_manager.config.dependencies))

    def test_install_dependencies_no_config(self):
        """Test dependency installation without configuration."""
        self.uv_manager.config = None

        result = self.uv_manager.install_dependencies()

        self.assertFalse(result)

    @patch('subprocess.run')
    def test_install_from_requirements_success(self, mock_run):
        """Test successful installation from requirements file."""
        requirements_path = self.temp_dir / "requirements.txt"
        requirements_path.write_text("numpy\nmatplotlib\n")

        mock_run.return_value = MagicMock(returncode=0)

        result = self.uv_manager.install_from_requirements(requirements_path)

        self.assertTrue(result)

    def test_install_from_requirements_file_not_found(self):
        """Test installation from non-existent requirements file."""
        requirements_path = self.temp_dir / "non_existent.txt"

        result = self.uv_manager.install_from_requirements(requirements_path)

        self.assertFalse(result)

    @patch('subprocess.run')
    def test_sync_environment_success(self, mock_run):
        """Test successful environment sync."""
        mock_run.return_value = MagicMock(returncode=0)

        result = self.uv_manager.sync_environment()

        self.assertTrue(result)

    @patch('subprocess.run')
    def test_run_uv_command_success(self, mock_run):
        """Test successful UV command execution."""
        mock_run.return_value = MagicMock(returncode=0, stdout="Command output")

        success, output = self.uv_manager.run_uv_command(["--version"])

        self.assertTrue(success)
        self.assertEqual(output, "Command output")

    @patch('subprocess.run')
    def test_run_uv_command_failure(self, mock_run):
        """Test failed UV command execution."""
        mock_run.return_value = MagicMock(returncode=1, stderr="Command failed")

        success, output = self.uv_manager.run_uv_command(["invalid-command"])

        self.assertFalse(success)
        self.assertEqual(output, "Command failed")

    def test_get_system_info(self):
        """Test getting comprehensive system information."""
        info = self.uv_manager.get_system_info()

        self.assertIn("platform", info)
        self.assertIn("python_version", info)
        self.assertIn("cpu_count", info)
        self.assertIn("memory_gb", info)
        self.assertIn("disk_free_gb", info)
        self.assertIn("uv_available", info)
        self.assertIn("uv_version", info)

        # Check that memory and disk values are reasonable
        self.assertGreater(info["memory_gb"], 0)
        self.assertGreater(info["disk_free_gb"], 0)

    def test_validate_environment_with_config(self):
        """Test environment validation with configuration."""
        self.uv_manager.config = self.uv_manager.create_default_config("test")

        with patch.object(self.uv_manager, 'check_uv_availability', return_value=(True, "uv 0.8.5")):
            validation = self.uv_manager.validate_environment()

        self.assertIn("valid", validation)
        self.assertIn("checks", validation)
        self.assertIn("python_version", validation["checks"])
        self.assertIn("memory", validation["checks"])
        self.assertIn("uv", validation["checks"])
        self.assertIn("visualization", validation["checks"])

    def test_validate_environment_no_config(self):
        """Test environment validation without configuration."""
        self.uv_manager.config = None

        validation = self.uv_manager.validate_environment()

        self.assertFalse(validation["valid"])
        self.assertIn("No configuration loaded", validation["error"])

    def test_setup_analytics_integration(self):
        """Test analytics integration setup."""
        result = self.uv_manager.setup_analytics_integration()

        self.assertTrue(result)

        # Check if analytics directory was created
        analytics_dir = self.uv_manager.extra_root / "analytics"
        self.assertTrue(analytics_dir.exists())

        # Check if config file was created
        config_file = analytics_dir / "analytics_config.json"
        self.assertTrue(config_file.exists())

        # Check config content
        with open(config_file, 'r') as f:
            config = json.load(f)

        self.assertIn("enabled", config)
        self.assertIn("backends", config)
        self.assertIn("metrics", config)

    def test_create_interactive_dashboard_setup(self):
        """Test interactive dashboard setup."""
        result = self.uv_manager.create_interactive_dashboard_setup()

        self.assertTrue(result)

        # Check if dashboards directory was created
        dashboard_dir = self.uv_manager.extra_root / "dashboards"
        self.assertTrue(dashboard_dir.exists())

        # Check if config file was created
        config_file = dashboard_dir / "dashboard_config.json"
        self.assertTrue(config_file.exists())

        # Check config content
        with open(config_file, 'r') as f:
            config = json.load(f)

        self.assertIn("enabled", config)
        self.assertIn("port", config)
        self.assertIn("components", config)

    def test_create_uv_script(self):
        """Test UV script creation."""
        content = "#!/usr/bin/env python3\nprint('Hello, UV!')"
        script_path = self.uv_manager.create_uv_script("test_script", content)

        self.assertTrue(script_path.exists())

        # Check script content
        with open(script_path, 'r') as f:
            script_content = f.read()

        self.assertIn("#!/usr/bin/env python3", script_content)
        self.assertIn("print('Hello, UV!')", script_content)

        # Check if script is executable
        self.assertTrue(os.access(script_path, os.X_OK))

    def test_export_environment_info(self):
        """Test environment information export."""
        self.uv_manager.config = self.uv_manager.create_default_config("test")

        output_path = self.uv_manager.export_environment_info()

        self.assertTrue(output_path.exists())

        # Check exported content
        with open(output_path, 'r') as f:
            info = json.load(f)

        self.assertIn("system_info", info)
        self.assertIn("validation_results", info)
        self.assertIn("configuration", info)


class TestUVConfig(unittest.TestCase):
    """Test cases for UV Configuration."""

    def test_uv_config_creation(self):
        """Test UV Config creation."""
        config = UVConfig(
            name="test_config",
            python_version="3.10",
            dependencies=["numpy", "matplotlib"],
            visualization_support=True
        )

        self.assertEqual(config.name, "test_config")
        self.assertEqual(config.python_version, "3.10")
        self.assertEqual(config.dependencies, ["numpy", "matplotlib"])
        self.assertTrue(config.visualization_support)

    def test_uv_config_to_dict(self):
        """Test converting UV Config to dictionary."""
        config = UVConfig(
            name="test_config",
            dependencies=["numpy"],
            visualization_support=True
        )

        config_dict = config.to_dict()

        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["name"], "test_config")
        self.assertEqual(config_dict["dependencies"], ["numpy"])
        self.assertTrue(config_dict["visualization_support"])

    def test_uv_config_from_dict(self):
        """Test creating UV Config from dictionary."""
        config_data = {
            "name": "test_config",
            "python_version": "3.10",
            "dependencies": ["numpy", "matplotlib"],
            "visualization_support": True
        }

        config = UVConfig.from_dict(config_data)

        self.assertIsInstance(config, UVConfig)
        self.assertEqual(config.name, "test_config")
        self.assertEqual(config.python_version, "3.10")
        self.assertEqual(config.dependencies, ["numpy", "matplotlib"])
        self.assertTrue(config.visualization_support)


if __name__ == '__main__':
    unittest.main()
