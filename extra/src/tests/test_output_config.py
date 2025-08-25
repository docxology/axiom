"""
Tests for Output Configuration System

Comprehensive tests for the OutputConfigManager and OutputPaths classes.
"""

import tempfile
import unittest
from pathlib import Path

# Add the src directory to the path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from output_config import OutputConfigManager, OutputPaths


class TestOutputPaths(unittest.TestCase):
    """Tests for OutputPaths class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_path = Path(self.temp_dir) / "test_output"
        self.paths = OutputPaths(base_dir=self.base_path)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test OutputPaths initialization."""
        self.assertEqual(self.paths.base_dir, self.base_path)
        self.assertTrue(str(self.paths.environment_configs).endswith("environment/configs"))
        self.assertTrue(str(self.paths.model_analysis).endswith("models/analysis"))
        self.assertTrue(str(self.paths.static_plots).endswith("visualizations/static"))

    def test_ensure_directories_exist(self):
        """Test creating all output directories."""
        self.paths.ensure_directories_exist()

        # Check that key directories were created
        self.assertTrue(self.paths.environment_configs.exists())
        self.assertTrue(self.paths.model_analysis.exists())
        self.assertTrue(self.paths.static_plots.exists())
        self.assertTrue(self.paths.performance_benchmarks.exists())

    def test_get_path_for_category(self):
        """Test getting paths for different categories."""
        # Test with category only
        path = self.paths.get_path_for_category("environment")
        self.assertEqual(path, self.paths.environment_configs)

        # Test with subcategory
        path = self.paths.get_path_for_category("environment", "reports")
        self.assertEqual(path, self.paths.environment_reports)

        # Test with invalid category
        with self.assertRaises(ValueError):
            self.paths.get_path_for_category("invalid_category")

        # Test with invalid subcategory (should return category default)
        path = self.paths.get_path_for_category("environment", "invalid_subcategory")
        self.assertEqual(path, self.paths.environment_configs)

    def test_set_base_dir(self):
        """Test updating base directory."""
        new_base = Path(self.temp_dir) / "new_output"
        self.paths.set_base_dir(new_base)

        self.assertEqual(self.paths.base_dir, new_base)
        self.assertTrue(str(self.paths.environment_configs).startswith(str(new_base)))

    def test_to_dict(self):
        """Test converting to dictionary."""
        config_dict = self.paths.to_dict()

        self.assertIn("base_dir", config_dict)
        self.assertIn("environment_configs", config_dict)
        self.assertIn("model_analysis", config_dict)
        self.assertIn("static_plots", config_dict)

        # Check that all paths are strings
        for key, value in config_dict.items():
            self.assertIsInstance(value, str)


class TestOutputConfigManager(unittest.TestCase):
    """Tests for OutputConfigManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_path = Path(self.temp_dir) / "test_output"
        # Reset singleton state for each test
        OutputConfigManager._instance = None
        OutputConfigManager._paths = None

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_singleton_pattern(self):
        """Test that OutputConfigManager follows singleton pattern."""
        manager1 = OutputConfigManager(self.base_path)
        manager2 = OutputConfigManager()

        # Should be the same instance
        self.assertIs(manager1, manager2)
        self.assertEqual(manager1.paths.base_dir, manager2.paths.base_dir)

    def test_initialization_with_base_dir(self):
        """Test initialization with custom base directory."""
        manager = OutputConfigManager(self.base_path)
        self.assertEqual(manager.paths.base_dir, self.base_path)

    def test_get_path_for(self):
        """Test getting paths through the manager."""
        manager = OutputConfigManager(self.base_path)

        path = manager.get_path_for("models", "analysis")
        self.assertEqual(path, manager.paths.model_analysis)

        path = manager.get_path_for("visualizations", "static")
        self.assertEqual(path, manager.paths.static_plots)

    def test_create_module_logger(self):
        """Test creating a module-specific logger."""
        manager = OutputConfigManager(self.base_path)

        logger = manager.create_module_logger("test_module")

        self.assertEqual(logger.name, "test_module")

        # Check that log directory was created
        log_dir = manager.paths.logs / "test_module"
        self.assertTrue(log_dir.exists())

    def test_set_base_dir_through_manager(self):
        """Test updating base directory through manager."""
        manager = OutputConfigManager(self.base_path)
        original_base = manager.paths.base_dir

        new_base = Path(self.temp_dir) / "new_output_base"
        manager.set_base_dir(new_base)

        self.assertEqual(manager.paths.base_dir, new_base)
        self.assertNotEqual(manager.paths.base_dir, original_base)


if __name__ == '__main__':
    unittest.main()
