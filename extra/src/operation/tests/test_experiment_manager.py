"""
Tests for Experiment Manager

Comprehensive tests for the ExperimentManager class.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the src directory to the path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from operation.experiment_manager import ExperimentManager, ExperimentConfig, ExperimentResult
from operation.model_runner import RunConfig, RunResult


class TestExperimentConfig(unittest.TestCase):
    """Tests for ExperimentConfig dataclass."""

    def test_experiment_config_creation(self):
        """Test creating an experiment configuration."""
        run_config = RunConfig(model_type="test", game_name="test")

        config = ExperimentConfig(
            name="test_experiment",
            description="Test experiment",
            model_configs=[{"model_type": "rmm"}, {"model_type": "hmm"}],
            game_names=["game1", "game2"],
            num_runs_per_config=3,
            run_config_template=run_config
        )

        self.assertEqual(config.name, "test_experiment")
        self.assertEqual(config.description, "Test experiment")
        self.assertEqual(len(config.model_configs), 2)
        self.assertEqual(len(config.game_names), 2)
        self.assertEqual(config.num_runs_per_config, 3)
        self.assertEqual(config.parallel_execution, False)
        self.assertEqual(config.max_workers, 4)

    def test_experiment_config_default_template(self):
        """Test experiment config with default template."""
        config = ExperimentConfig(
            name="test_experiment",
            description="Test experiment",
            model_configs=[{"model_type": "rmm"}],
            game_names=["game1"],
            num_runs_per_config=1
        )

        # Should have default run config template
        self.assertIsNotNone(config.run_config_template)
        self.assertEqual(config.run_config_template.num_steps, 1000)
        self.assertEqual(config.run_config_template.num_episodes, 5)


class TestExperimentResult(unittest.TestCase):
    """Tests for ExperimentResult dataclass."""

    def test_experiment_result_creation(self):
        """Test creating an experiment result."""
        run_results = [
            RunResult(
                run_id="run_001",
                model_type="rmm",
                game_name="game1",
                total_steps=1000,
                total_episodes=5,
                total_reward=500.0,
                average_reward=100.0,
                best_reward=150.0,
                worst_reward=50.0,
                duration_seconds=120.5,
                success=True
            )
        ]

        result = ExperimentResult(
            experiment_name="test_experiment",
            total_runs=5,
            successful_runs=4,
            failed_runs=1,
            run_results=run_results,
            best_run=run_results[0],
            worst_run=run_results[0],
            average_performance={"average_reward": 100.0},
            duration_seconds=600.0,
            start_time=1000000.0,
            end_time=1000600.0
        )

        self.assertEqual(result.experiment_name, "test_experiment")
        self.assertEqual(result.total_runs, 5)
        self.assertEqual(result.successful_runs, 4)
        self.assertEqual(result.failed_runs, 1)
        self.assertEqual(len(result.run_results), 1)
        self.assertEqual(result.duration_seconds, 600.0)


class TestExperimentManager(unittest.TestCase):
    """Tests for ExperimentManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = ExperimentManager()

    def test_initialization(self):
        """Test ExperimentManager initialization."""
        manager = ExperimentManager()
        self.assertIsNotNone(manager.model_runner)
        self.assertIsNotNone(manager.base_dir)

    def test_initialization_with_base_dir(self):
        """Test ExperimentManager initialization with custom base directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManager(temp_dir)
            self.assertTrue(Path(temp_dir).exists())

    def test_list_experiments_empty_directory(self):
        """Test listing experiments in empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManager(temp_dir)
            experiments = manager.list_experiments()
            self.assertEqual(experiments, [])

    def test_load_experiment_result_nonexistent(self):
        """Test loading non-existent experiment result."""
        result = self.manager.load_experiment_result("nonexistent_experiment")
        self.assertIsNone(result)

    def test_save_and_load_experiment_result(self):
        """Test saving and loading experiment results."""
        run_results = [
            RunResult(
                run_id="run_001",
                model_type="rmm",
                game_name="game1",
                total_steps=1000,
                total_episodes=5,
                total_reward=500.0,
                average_reward=100.0,
                best_reward=150.0,
                worst_reward=50.0,
                duration_seconds=120.5,
                success=True
            )
        ]

        experiment_result = ExperimentResult(
            experiment_name="test_experiment",
            total_runs=1,
            successful_runs=1,
            failed_runs=0,
            run_results=run_results,
            best_run=run_results[0],
            worst_run=run_results[0],
            average_performance={"average_reward": 100.0},
            duration_seconds=120.5,
            start_time=1000000.0,
            end_time=1000120.5
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManager(temp_dir)

            # Save result
            manager.save_experiment_result(experiment_result)

            # Load result
            loaded_result = manager.load_experiment_result("test_experiment")

            self.assertIsNotNone(loaded_result)
            self.assertEqual(loaded_result.experiment_name, experiment_result.experiment_name)
            self.assertEqual(loaded_result.total_runs, experiment_result.total_runs)
            self.assertEqual(loaded_result.successful_runs, experiment_result.successful_runs)

    def test_calculate_average_performance_empty_list(self):
        """Test calculating average performance with empty list."""
        performance = self.manager._calculate_average_performance([])
        expected_keys = [
            "average_reward", "best_reward", "worst_reward",
            "total_reward", "average_duration"
        ]

        for key in expected_keys:
            self.assertIn(key, performance)
            self.assertEqual(performance[key], 0.0)

    def test_calculate_average_performance_with_data(self):
        """Test calculating average performance with data."""
        run_results = [
            RunResult(
                run_id="run_001",
                model_type="rmm",
                game_name="game1",
                total_steps=1000,
                total_episodes=5,
                total_reward=500.0,
                average_reward=100.0,
                best_reward=150.0,
                worst_reward=50.0,
                duration_seconds=120.5,
                success=True
            ),
            RunResult(
                run_id="run_002",
                model_type="rmm",
                game_name="game1",
                total_steps=1000,
                total_episodes=5,
                total_reward=600.0,
                average_reward=120.0,
                best_reward=180.0,
                worst_reward=60.0,
                duration_seconds=110.5,
                success=True
            )
        ]

        performance = self.manager._calculate_average_performance(run_results)

        self.assertAlmostEqual(performance["average_reward"], 110.0)
        self.assertEqual(performance["best_reward"], 180.0)
        self.assertEqual(performance["worst_reward"], 50.0)
        self.assertEqual(performance["total_reward"], 1100.0)
        self.assertAlmostEqual(performance["average_duration"], 115.5)

    def test_run_sequential_experiment(self):
        """Test running sequential experiment (mocked)."""
        config = ExperimentConfig(
            name="test_experiment",
            description="Test experiment",
            model_configs=[{"model_type": "rmm"}],
            game_names=["game1"],
            num_runs_per_config=2,
            parallel_execution=False
        )

        # Mock the model runner
        mock_run_result = RunResult(
            run_id="mock_run",
            model_type="rmm",
            game_name="game1",
            total_steps=100,
            total_episodes=1,
            total_reward=50.0,
            average_reward=50.0,
            best_reward=50.0,
            worst_reward=50.0,
            duration_seconds=10.0,
            success=True
        )

        with patch.object(self.manager.model_runner, 'run_model') as mock_run:
            mock_run.return_value = mock_run_result

            def mock_model_factory(model_config):
                return MagicMock()

            def mock_game_factory(game_name):
                return MagicMock()

            result = self.manager.run_experiment(
                config, mock_model_factory, mock_game_factory
            )

            self.assertIsInstance(result, ExperimentResult)
            self.assertEqual(result.experiment_name, "test_experiment")
            self.assertEqual(result.total_runs, 2)  # 1 config * 1 game * 2 runs
            self.assertEqual(result.successful_runs, 2)
            self.assertEqual(result.failed_runs, 0)

            # Verify run_model was called the expected number of times
            self.assertEqual(mock_run.call_count, 2)

    def test_run_experiment_with_failures(self):
        """Test running experiment with some failures."""
        config = ExperimentConfig(
            name="test_experiment",
            description="Test experiment",
            model_configs=[{"model_type": "rmm"}],
            game_names=["game1"],
            num_runs_per_config=3,
            parallel_execution=False
        )

        # Create mock results - some successful, some failed
        successful_result = RunResult(
            run_id="mock_run_success",
            model_type="rmm",
            game_name="game1",
            total_steps=100,
            total_episodes=1,
            total_reward=50.0,
            average_reward=50.0,
            best_reward=50.0,
            worst_reward=50.0,
            duration_seconds=10.0,
            success=True
        )

        failed_result = RunResult(
            run_id="mock_run_failed",
            model_type="rmm",
            game_name="game1",
            total_steps=0,
            total_episodes=0,
            total_reward=0.0,
            average_reward=0.0,
            best_reward=0.0,
            worst_reward=0.0,
            duration_seconds=5.0,
            success=False,
            error_message="Mock failure"
        )

        with patch.object(self.manager.model_runner, 'run_model') as mock_run:
            mock_run.side_effect = [successful_result, failed_result, successful_result]

            def mock_model_factory(model_config):
                return MagicMock()

            def mock_game_factory(game_name):
                return MagicMock()

            result = self.manager.run_experiment(
                config, mock_model_factory, mock_game_factory
            )

            self.assertEqual(result.total_runs, 3)
            self.assertEqual(result.successful_runs, 2)
            self.assertEqual(result.failed_runs, 1)

    def test_compare_experiments_no_experiments(self):
        """Test comparing experiments when none exist."""
        comparison = self.manager.compare_experiments([])
        self.assertIn("error", comparison)

    def test_compare_experiments_with_data(self):
        """Test comparing experiments with actual data."""
        # Create mock experiment results
        run_results_1 = [
            RunResult(
                run_id="run_001",
                model_type="rmm",
                game_name="game1",
                total_steps=100,
                total_episodes=1,
                total_reward=100.0,
                average_reward=100.0,
                best_reward=100.0,
                worst_reward=100.0,
                duration_seconds=10.0,
                success=True
            )
        ]

        run_results_2 = [
            RunResult(
                run_id="run_002",
                model_type="hmm",
                game_name="game1",
                total_steps=100,
                total_episodes=1,
                total_reward=80.0,
                average_reward=80.0,
                best_reward=80.0,
                worst_reward=80.0,
                duration_seconds=10.0,
                success=True
            )
        ]

        exp_result_1 = ExperimentResult(
            experiment_name="exp1",
            total_runs=1,
            successful_runs=1,
            failed_runs=0,
            run_results=run_results_1,
            best_run=run_results_1[0],
            worst_run=run_results_1[0],
            average_performance={
                "average_reward": 100.0,
                "best_reward": 100.0,
                "worst_reward": 100.0,
                "total_reward": 100.0,
                "average_duration": 10.0
            },
            duration_seconds=10.0,
            start_time=1000000.0,
            end_time=1000010.0
        )

        exp_result_2 = ExperimentResult(
            experiment_name="exp2",
            total_runs=1,
            successful_runs=1,
            failed_runs=0,
            run_results=run_results_2,
            best_run=run_results_2[0],
            worst_run=run_results_2[0],
            average_performance={
                "average_reward": 80.0,
                "best_reward": 80.0,
                "worst_reward": 80.0,
                "total_reward": 80.0,
                "average_duration": 10.0
            },
            duration_seconds=10.0,
            start_time=1000010.0,
            end_time=1000020.0
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManager(temp_dir)

            # Save experiment results
            manager.save_experiment_result(exp_result_1)
            manager.save_experiment_result(exp_result_2)

            # Compare experiments
            comparison = manager.compare_experiments(["exp1", "exp2"])

            self.assertIn("best_experiment", comparison)
            self.assertIn("worst_experiment", comparison)
            self.assertIn("performance_ranking", comparison)
            self.assertEqual(comparison["best_experiment"].experiment_name, "exp1")
            self.assertEqual(comparison["worst_experiment"].experiment_name, "exp2")
            self.assertEqual(len(comparison["performance_ranking"]), 2)

    def test_generate_experiment_report_nonexistent(self):
        """Test generating report for non-existent experiment."""
        report = self.manager.generate_experiment_report("nonexistent_experiment")
        self.assertIn("not found", report)

    def test_generate_experiment_report_with_data(self):
        """Test generating experiment report with data."""
        run_results = [
            RunResult(
                run_id="run_001",
                model_type="rmm",
                game_name="game1",
                total_steps=100,
                total_episodes=1,
                total_reward=100.0,
                average_reward=100.0,
                best_reward=100.0,
                worst_reward=100.0,
                duration_seconds=10.0,
                success=True
            )
        ]

        exp_result = ExperimentResult(
            experiment_name="test_exp",
            total_runs=1,
            successful_runs=1,
            failed_runs=0,
            run_results=run_results,
            best_run=run_results[0],
            worst_run=run_results[0],
            average_performance={
                "average_reward": 100.0,
                "best_reward": 100.0,
                "worst_reward": 100.0,
                "total_reward": 100.0,
                "average_duration": 10.0
            },
            duration_seconds=10.0,
            start_time=1000000.0,
            end_time=1000010.0
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManager(temp_dir)
            manager.save_experiment_result(exp_result)

            report = manager.generate_experiment_report("test_exp")

            self.assertIn("EXPERIMENT REPORT: test_exp", report)
            self.assertIn("Total Runs: 1", report)
            self.assertIn("Successful Runs: 1", report)
            self.assertIn("Average Reward: 100.00", report)


if __name__ == '__main__':
    unittest.main()
