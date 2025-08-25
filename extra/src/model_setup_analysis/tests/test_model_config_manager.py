"""
Tests for Model Configuration Manager

Comprehensive tests for the ModelConfigManager class.
"""

import json
import tempfile
import unittest
from pathlib import Path

# Add the src directory to the path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model_setup_analysis.model_config_manager import ModelConfigManager, ModelPreset


class TestModelPreset(unittest.TestCase):
    """Tests for ModelPreset dataclass."""

    def test_preset_creation(self):
        """Test creating a model preset."""
        preset = ModelPreset(
            name="test_preset",
            model_type="rmm",
            description="Test preset",
            config={"hidden_dim": 256},
            performance_target="balanced",
            memory_efficient=False
        )

        self.assertEqual(preset.name, "test_preset")
        self.assertEqual(preset.model_type, "rmm")
        self.assertEqual(preset.description, "Test preset")
        self.assertEqual(preset.config, {"hidden_dim": 256})
        self.assertEqual(preset.performance_target, "balanced")
        self.assertEqual(preset.memory_efficient, False)


class TestModelConfigManager(unittest.TestCase):
    """Tests for ModelConfigManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = ModelConfigManager()

    def test_initialization(self):
        """Test ModelConfigManager initialization."""
        manager = ModelConfigManager()
        self.assertIsNotNone(manager._presets)

        # Should have default presets
        self.assertGreater(len(manager._presets), 0)

    def test_get_preset(self):
        """Test getting a preset."""
        preset = self.manager.get_preset("rmm_light")
        self.assertIsNotNone(preset)
        self.assertEqual(preset.name, "rmm_light")
        self.assertEqual(preset.model_type, "rmm")

    def test_get_preset_nonexistent(self):
        """Test getting a non-existent preset."""
        preset = self.manager.get_preset("nonexistent")
        self.assertIsNone(preset)

    def test_list_presets(self):
        """Test listing all presets."""
        presets = self.manager.list_presets()
        self.assertGreater(len(presets), 0)

        # Should contain our default presets
        preset_names = [p.name for p in presets]
        self.assertIn("rmm_light", preset_names)
        self.assertIn("hmm_efficient", preset_names)

    def test_list_presets_by_type(self):
        """Test listing presets filtered by model type."""
        rmm_presets = self.manager.list_presets("rmm")
        hmm_presets = self.manager.list_presets("hmm")

        self.assertGreater(len(rmm_presets), 0)
        self.assertGreater(len(hmm_presets), 0)

        # Check that all returned presets have the correct type
        for preset in rmm_presets:
            self.assertEqual(preset.model_type, "rmm")

        for preset in hmm_presets:
            self.assertEqual(preset.model_type, "hmm")

    def test_create_preset(self):
        """Test creating a new preset."""
        new_preset = ModelPreset(
            name="custom_preset",
            model_type="rmm",
            description="Custom preset",
            config={"hidden_dim": 512},
            performance_target="accuracy"
        )

        self.manager.create_preset(new_preset)

        # Verify it was added
        retrieved = self.manager.get_preset("custom_preset")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "custom_preset")
        self.assertEqual(retrieved.config["hidden_dim"], 512)

    def test_save_and_load_preset(self):
        """Test saving and loading presets."""
        preset = ModelPreset(
            name="test_save",
            model_type="rmm",
            description="Test save preset",
            config={"hidden_dim": 128}
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Save preset
            self.manager.save_preset(preset, temp_path)

            # Load preset
            loaded_preset = self.manager.load_preset(temp_path)

            self.assertEqual(loaded_preset.name, preset.name)
            self.assertEqual(loaded_preset.model_type, preset.model_type)
            self.assertEqual(loaded_preset.config, preset.config)

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_load_preset_nonexistent(self):
        """Test loading a non-existent preset file."""
        with self.assertRaises(FileNotFoundError):
            self.manager.load_preset("/nonexistent/path.json")

    def test_generate_optimized_config_rmm(self):
        """Test generating optimized RMM configuration."""
        config = self.manager.generate_optimized_config("rmm", "balanced")

        required_fields = ["num_slots", "slot_size", "hidden_dim", "num_objects", "learning_rate"]
        for field in required_fields:
            self.assertIn(field, config)

        # Check that values are reasonable
        self.assertGreater(config["num_slots"], 0)
        self.assertGreater(config["hidden_dim"], 0)
        self.assertGreater(config["learning_rate"], 0)

    def test_generate_optimized_config_hmm(self):
        """Test generating optimized HMM configuration."""
        config = self.manager.generate_optimized_config("hmm", "balanced")

        required_fields = ["num_layers", "hidden_dim", "memory_size", "attention_heads", "learning_rate"]
        for field in required_fields:
            self.assertIn(field, config)

        # Check that values are reasonable
        self.assertGreater(config["num_layers"], 0)
        self.assertGreater(config["hidden_dim"], 0)
        self.assertGreater(config["attention_heads"], 0)

    def test_generate_optimized_config_speed(self):
        """Test generating speed-optimized configuration."""
        config = self.manager.generate_optimized_config("rmm", "speed")

        # Speed-optimized should have smaller values
        self.assertLessEqual(config["num_slots"], 10)
        self.assertLessEqual(config["hidden_dim"], 256)

    def test_generate_optimized_config_accuracy(self):
        """Test generating accuracy-optimized configuration."""
        config = self.manager.generate_optimized_config("hmm", "accuracy")

        # Accuracy-optimized should have larger values
        self.assertGreaterEqual(config["num_layers"], 3)
        self.assertGreaterEqual(config["hidden_dim"], 256)

    def test_generate_optimized_config_unsupported_type(self):
        """Test generating config for unsupported model type."""
        with self.assertRaises(ValueError):
            self.manager.generate_optimized_config("unsupported_type")

    def test_validate_config_rmm(self):
        """Test validating RMM configuration."""
        valid_config = {
            "num_slots": 10,
            "slot_size": 64,
            "hidden_dim": 256,
            "num_objects": 5,
            "learning_rate": 0.001
        }

        errors = self.manager.validate_config(valid_config, "rmm")
        self.assertEqual(len(errors), 0)

    def test_validate_config_rmm_missing_fields(self):
        """Test validating RMM configuration with missing fields."""
        invalid_config = {"hidden_dim": 256}  # Missing required fields

        errors = self.manager.validate_config(invalid_config, "rmm")
        self.assertGreater(len(errors), 0)

        # Should contain errors about missing fields
        error_text = " ".join(errors)
        self.assertIn("num_slots", error_text)
        self.assertIn("slot_size", error_text)

    def test_validate_config_hmm(self):
        """Test validating HMM configuration."""
        valid_config = {
            "num_layers": 3,
            "hidden_dim": 256,
            "memory_size": 128,
            "attention_heads": 8,
            "learning_rate": 0.001
        }

        errors = self.manager.validate_config(valid_config, "hmm")
        self.assertEqual(len(errors), 0)

    def test_validate_config_invalid_values(self):
        """Test validating configuration with invalid values."""
        invalid_config = {
            "num_slots": -1,  # Invalid negative value
            "slot_size": 64,
            "hidden_dim": 0,  # Invalid zero value
            "num_objects": 5,
            "learning_rate": 1.5  # Invalid learning rate > 1
        }

        errors = self.manager.validate_config(invalid_config, "rmm")
        self.assertGreater(len(errors), 0)

        # Should contain errors about invalid values
        error_text = " ".join(errors)
        self.assertIn("hidden_dim", error_text)
        self.assertIn("learning_rate", error_text)

    def test_validate_config_unsupported_type(self):
        """Test validating configuration for unsupported model type."""
        config = {"hidden_dim": 256}
        errors = self.manager.validate_config(config, "unsupported")

        self.assertGreater(len(errors), 0)
        self.assertIn("Unsupported model type", errors[0])

    def test_interpolate_configs(self):
        """Test interpolating between configurations."""
        config1 = {"hidden_dim": 128, "learning_rate": 0.001, "num_layers": 2}
        config2 = {"hidden_dim": 512, "learning_rate": 0.0001, "num_layers": 6}

        # Test midpoint interpolation
        interpolated = self.manager.interpolate_configs(config1, config2, 0.5)

        self.assertEqual(interpolated["hidden_dim"], 320.0)  # Midpoint
        self.assertAlmostEqual(interpolated["learning_rate"], 0.00055, places=5)  # Midpoint
        self.assertEqual(interpolated["num_layers"], 4.0)  # Midpoint

    def test_interpolate_configs_edge_cases(self):
        """Test interpolating with edge case ratios."""
        config1 = {"hidden_dim": 128}
        config2 = {"hidden_dim": 512}

        # Test ratio 0 (should return config1)
        interpolated_0 = self.manager.interpolate_configs(config1, config2, 0.0)
        self.assertEqual(interpolated_0["hidden_dim"], 128)

        # Test ratio 1 (should return config2)
        interpolated_1 = self.manager.interpolate_configs(config1, config2, 1.0)
        self.assertEqual(interpolated_1["hidden_dim"], 512)

    def test_interpolate_configs_invalid_ratio(self):
        """Test interpolating with invalid ratio."""
        config1 = {"hidden_dim": 128}
        config2 = {"hidden_dim": 512}

        with self.assertRaises(ValueError):
            self.manager.interpolate_configs(config1, config2, 1.5)

        with self.assertRaises(ValueError):
            self.manager.interpolate_configs(config1, config2, -0.5)

    def test_interpolate_configs_mixed_types(self):
        """Test interpolating configs with mixed value types."""
        config1 = {"hidden_dim": 128, "model_type": "rmm", "use_bias": True}
        config2 = {"hidden_dim": 512, "model_type": "hmm", "use_bias": False}

        interpolated = self.manager.interpolate_configs(config1, config2, 0.5)

        # Numeric values should be interpolated
        self.assertEqual(interpolated["hidden_dim"], 320.0)

        # Non-numeric values should use config1
        self.assertEqual(interpolated["model_type"], "rmm")
        # Boolean values are treated as numeric (True=1, False=0), so they get interpolated
        self.assertEqual(interpolated["use_bias"], 0.5)  # Midpoint between True (1) and False (0)

    def test_export_config_summary(self):
        """Test exporting configuration summary."""
        config = {
            "hidden_dim": 256,
            "learning_rate": 0.001,
            "num_slots": 10,
            "slot_size": 64,
            "num_objects": 5
        }

        summary = self.manager.export_config_summary(config, "rmm")

        # Check that summary contains expected sections
        self.assertIn("Model Configuration Summary", summary)
        self.assertIn("Model Type: rmm", summary)
        self.assertIn("hidden_dim", summary)
        self.assertIn("num_slots", summary)
        self.assertIn("slot_size", summary)

    def test_estimate_memory_usage(self):
        """Test memory usage estimation."""
        config = {
            "num_slots": 10,
            "slot_size": 64,
            "hidden_dim": 256,
            "num_objects": 5
        }

        # Test RMM memory estimation
        memory = self.manager._estimate_memory_usage(config, "rmm")
        self.assertGreater(memory, 0)

        # Test HMM memory estimation
        hmm_config = {
            "memory_size": 128,
            "hidden_dim": 256,
            "num_layers": 3
        }
        hmm_memory = self.manager._estimate_memory_usage(hmm_config, "hmm")
        self.assertGreater(hmm_memory, 0)


if __name__ == '__main__':
    unittest.main()
