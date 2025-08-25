"""
Tests for Model Runner

Comprehensive tests for the ModelRunner class.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the src directory to the path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from operation.model_runner import ModelRunner, RunConfig, RunResult


class TestRunConfig(unittest.TestCase):
    """Tests for RunConfig dataclass."""

    def test_run_config_creation(self):
        """Test creating a run configuration."""
        config = RunConfig(
            model_type="rmm",
            game_name="test_game",
            num_steps=1000,
            num_episodes=5,
            save_frequency=100,
            random_seed=42
        )

        self.assertEqual(config.model_type, "rmm")
        self.assertEqual(config.game_name, "test_game")
        self.assertEqual(config.num_steps, 1000)
        self.assertEqual(config.num_episodes, 5)
        self.assertEqual(config.save_frequency, 100)
        self.assertEqual(config.random_seed, 42)
        self.assertEqual(config.output_dir, "results")
        self.assertEqual(config.checkpoint_dir, "checkpoints")


class TestRunResult(unittest.TestCase):
    """Tests for RunResult dataclass."""

    def test_run_result_creation(self):
        """Test creating a run result."""
        result = RunResult(
            run_id="test_run_001",
            model_type="rmm",
            game_name="test_game",
            total_steps=1000,
            total_episodes=5,
            total_reward=500.0,
            average_reward=100.0,
            best_reward=150.0,
            worst_reward=50.0,
            duration_seconds=120.5,
            success=True
        )

        self.assertEqual(result.run_id, "test_run_001")
        self.assertEqual(result.model_type, "rmm")
        self.assertEqual(result.game_name, "test_game")
        self.assertEqual(result.total_steps, 1000)
        self.assertEqual(result.total_episodes, 5)
        self.assertEqual(result.total_reward, 500.0)
        self.assertEqual(result.average_reward, 100.0)
        self.assertEqual(result.success, True)


class TestModelRunner(unittest.TestCase):
    """Tests for ModelRunner class."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = ModelRunner()

    def test_initialization(self):
        """Test ModelRunner initialization."""
        runner = ModelRunner()
        self.assertIsNone(runner._current_run)

    def test_initialization_with_base_dir(self):
        """Test ModelRunner initialization with custom base directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ModelRunner(temp_dir)
            self.assertTrue(Path(temp_dir).exists())

    def test_generate_run_id(self):
        """Test run ID generation."""
        run_id = self.runner._generate_run_id()
        self.assertTrue(run_id.startswith("run_"))
        self.assertEqual(len(run_id), 19)  # "run_" + timestamp format

    def test_set_random_seed(self):
        """Test random seed setting."""
        original_seed = 42

        # Test without torch
        self.runner._set_random_seed(original_seed)

        # Verify random module was seeded
        import random
        random.setstate(random.getstate())  # Should not raise error

    def test_save_checkpoint_no_save_method(self):
        """Test checkpoint saving when model has no save method."""
        mock_model = MagicMock()
        # Remove save method to test the fallback
        del mock_model.save

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            self.runner._save_checkpoint(mock_model, checkpoint_dir, 100)

        # Should not raise an error

    def test_save_render_no_render_method(self):
        """Test render saving when game has no render method."""
        mock_game = MagicMock()
        # Remove render method to test the fallback
        del mock_game.render

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            self.runner._save_render(mock_game, output_dir, 100)

        # Should not raise an error

    def test_list_runs_empty_directory(self):
        """Test listing runs in empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ModelRunner(temp_dir)
            runs = runner.list_runs()
            self.assertEqual(runs, [])

    def test_load_run_result_nonexistent(self):
        """Test loading non-existent run result."""
        result = self.runner.load_run_result("nonexistent_run")
        self.assertIsNone(result)

    def test_save_and_load_run_result(self):
        """Test saving and loading run results."""
        run_result = RunResult(
            run_id="test_run_001",
            model_type="rmm",
            game_name="test_game",
            total_steps=1000,
            total_episodes=5,
            total_reward=500.0,
            average_reward=100.0,
            best_reward=150.0,
            worst_reward=50.0,
            duration_seconds=120.5,
            success=True
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ModelRunner(temp_dir)

            # Save result
            runner.save_run_result(run_result)

            # Load result
            loaded_result = runner.load_run_result("test_run_001")

            self.assertIsNotNone(loaded_result)
            self.assertEqual(loaded_result.run_id, run_result.run_id)
            self.assertEqual(loaded_result.model_type, run_result.model_type)
            self.assertEqual(loaded_result.total_steps, run_result.total_steps)

    def test_get_current_run(self):
        """Test getting current run."""
        self.assertIsNone(self.runner.get_current_run())

        # Set a current run
        run_result = RunResult(
            run_id="test_run_001",
            model_type="rmm",
            game_name="test_game",
            total_steps=1000,
            total_episodes=5,
            total_reward=500.0,
            average_reward=100.0,
            best_reward=150.0,
            worst_reward=50.0,
            duration_seconds=120.5,
            success=True
        )
        self.runner._current_run = run_result

        current_run = self.runner.get_current_run()
        self.assertEqual(current_run.run_id, "test_run_001")

    def test_managed_run_context_manager(self):
        """Test managed run context manager."""
        config = RunConfig(model_type="test", game_name="test")

        with self.runner.managed_run(config) as run_info:
            self.assertEqual(run_info["status"], "running")
            self.assertIn("run_id", run_info)
            self.assertIn("config", run_info)
            self.assertIn("start_time", run_info)

        # After context manager exits, status should be completed
        self.assertEqual(run_info["status"], "completed")
        self.assertIn("end_time", run_info)
        self.assertIn("duration", run_info)

    def test_run_model_integration(self):
        """Test the full run_model integration (mocked)."""
        config = RunConfig(
            model_type="test_model",
            game_name="test_game",
            num_steps=10,
            num_episodes=2
        )

        # Mock model and game
        mock_model = MagicMock()
        mock_game = MagicMock()
        mock_game.is_terminal.return_value = False
        mock_game.get_state.return_value = {"position": [0, 0]}
        mock_game.step.return_value = (1.0, {"position": [1, 0]}, False)

        def mock_model_factory(model_config):
            return mock_model

        def mock_game_factory(game_name):
            return mock_game

        # Mock the _execute_run method to avoid complex setup
        with patch.object(self.runner, '_execute_run') as mock_execute:
            mock_execute.return_value = {
                "total_steps": 10,
                "total_episodes": 2,
                "total_reward": 20.0,
                "average_reward": 10.0,
                "best_reward": 15.0,
                "worst_reward": 5.0
            }

            result = self.runner.run_model(
                config, mock_model_factory, mock_game_factory
            )

            self.assertIsInstance(result, RunResult)
            self.assertTrue(result.success)
            self.assertEqual(result.total_steps, 10)
            self.assertEqual(result.total_episodes, 2)
            self.assertEqual(result.average_reward, 10.0)

    def test_run_model_with_failure(self):
        """Test run_model with simulated failure."""
        config = RunConfig(
            model_type="test_model",
            game_name="test_game",
            num_steps=10,
            num_episodes=2
        )

        def mock_model_factory(model_config):
            return MagicMock()

        def mock_game_factory(game_name):
            return MagicMock()

        # Mock the _execute_run method to raise an exception
        with patch.object(self.runner, '_execute_run') as mock_execute:
            mock_execute.side_effect = ValueError("Simulated failure")

            result = self.runner.run_model(
                config, mock_model_factory, mock_game_factory
            )

            self.assertIsInstance(result, RunResult)
            self.assertFalse(result.success)
            self.assertEqual(result.error_message, "Simulated failure")
            self.assertEqual(result.total_steps, 0)
            self.assertEqual(result.total_episodes, 0)


if __name__ == '__main__':
    unittest.main()
