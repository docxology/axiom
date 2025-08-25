"""
Tests for Model Analyzer

Comprehensive tests for the ModelAnalyzer class.
"""

import unittest
from pathlib import Path

# Add the src directory to the path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model_setup_analysis.model_analyzer import ModelAnalyzer


class TestModelAnalyzer(unittest.TestCase):
    """Tests for ModelAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ModelAnalyzer()
        self.rmm_config = {
            "model_type": "rmm",
            "num_slots": 10,
            "slot_size": 64,
            "hidden_dim": 256,
            "num_objects": 5,
            "learning_rate": 0.001
        }
        self.hmm_config = {
            "model_type": "hmm",
            "num_layers": 3,
            "hidden_dim": 256,
            "memory_size": 128,
            "attention_heads": 8,
            "learning_rate": 0.001
        }

    def test_initialization(self):
        """Test ModelAnalyzer initialization."""
        analyzer = ModelAnalyzer()
        self.assertEqual(analyzer.model_config, {})  # Default is empty dict, not None

        config = {"model_type": "rmm"}
        analyzer_with_config = ModelAnalyzer(config)
        self.assertEqual(analyzer_with_config.model_config, config)

    def test_analyze_rmm_complexity(self):
        """Test RMM complexity analysis."""
        analysis = self.analyzer.analyze_model_complexity(self.rmm_config)

        self.assertEqual(analysis["model_type"], "rmm")
        self.assertIn("complexity_score", analysis)
        self.assertIn("parameter_estimate", analysis)
        self.assertIn("memory_estimate_gb", analysis)
        self.assertIn("complexity_factors", analysis)

        # Check that complexity factors are present
        factors = analysis["complexity_factors"]
        self.assertIn("slot_complexity", factors)
        self.assertIn("object_complexity", factors)
        self.assertIn("hidden_dimension", factors)
        self.assertIn("relational_operations", factors)

        # Check values are reasonable
        self.assertGreater(analysis["complexity_score"], 0)
        self.assertGreater(analysis["parameter_estimate"], 0)
        self.assertGreater(analysis["memory_estimate_gb"], 0)

    def test_analyze_hmm_complexity(self):
        """Test HMM complexity analysis."""
        analysis = self.analyzer.analyze_model_complexity(self.hmm_config)

        self.assertEqual(analysis["model_type"], "hmm")
        self.assertIn("complexity_score", analysis)
        self.assertIn("parameter_estimate", analysis)
        self.assertIn("memory_estimate_gb", analysis)
        self.assertIn("complexity_factors", analysis)

        # Check that complexity factors are present
        factors = analysis["complexity_factors"]
        self.assertIn("attention_complexity", factors)
        self.assertIn("memory_complexity", factors)
        self.assertIn("layer_complexity", factors)
        self.assertIn("transformer_operations", factors)

    def test_analyze_unknown_model_type(self):
        """Test analysis of unknown model type."""
        config = {"model_type": "unknown"}
        analysis = self.analyzer.analyze_model_complexity(config)

        self.assertEqual(analysis["model_type"], "unknown")
        self.assertIn("complexity_factors", analysis)
        self.assertIn("unknown_model_type", analysis["complexity_factors"])

    def test_analyze_no_config(self):
        """Test analysis with no configuration."""
        analysis = self.analyzer.analyze_model_complexity()

        self.assertIn("error", analysis)
        self.assertEqual(analysis["error"], "No model configuration provided")

    def test_compare_model_configs(self):
        """Test comparing two model configurations."""
        comparison = self.analyzer.compare_model_configs(self.rmm_config, self.hmm_config)

        self.assertIn("config1_type", comparison)
        self.assertIn("config2_type", comparison)
        self.assertIn("complexity_comparison", comparison)
        self.assertIn("parameter_comparison", comparison)
        self.assertIn("memory_comparison", comparison)

        self.assertEqual(comparison["config1_type"], "rmm")
        self.assertEqual(comparison["config2_type"], "hmm")

        # Check comparison values
        comp = comparison["complexity_comparison"]
        self.assertIn("config1_score", comp)
        self.assertIn("config2_score", comp)
        self.assertIn("complexity_difference", comp)

    def test_identify_optimization_opportunities_high_complexity(self):
        """Test identifying optimization opportunities for high complexity model."""
        high_complexity_config = {
            "model_type": "rmm",
            "num_slots": 1000,  # Extremely high number of slots
            "slot_size": 1000,  # Extremely large slot size
            "hidden_dim": 10000,  # Extremely high
            "num_objects": 1000,  # Extremely high number of objects
            "learning_rate": 0.001
        }

        opportunities = self.analyzer.identify_optimization_opportunities(high_complexity_config)

        # Should identify high complexity issues
        self.assertTrue(len(opportunities) > 0)

        # Check that high complexity opportunity is identified
        high_complexity_found = any("high_complexity" in opp.get("type", "") for opp in opportunities)
        self.assertTrue(high_complexity_found)

    def test_identify_optimization_opportunities_rmm_slots(self):
        """Test RMM-specific optimization opportunities."""
        rmm_config_high_slots = {
            "model_type": "rmm",
            "num_slots": 30,  # High number of slots
            "slot_size": 64,
            "hidden_dim": 256,
            "num_objects": 5,
            "learning_rate": 0.001
        }

        opportunities = self.analyzer.identify_optimization_opportunities(rmm_config_high_slots)

        # Should identify RMM slot optimization opportunities
        rmm_slot_found = any("rmm_slot_optimization" in opp.get("type", "") for opp in opportunities)
        self.assertTrue(rmm_slot_found)

    def test_identify_optimization_opportunities_hmm_attention(self):
        """Test HMM-specific optimization opportunities."""
        hmm_config_high_attention = {
            "model_type": "hmm",
            "num_layers": 3,
            "hidden_dim": 256,
            "memory_size": 128,
            "attention_heads": 16,  # High number of attention heads
            "learning_rate": 0.001
        }

        opportunities = self.analyzer.identify_optimization_opportunities(hmm_config_high_attention)

        # Should identify HMM attention optimization opportunities
        hmm_attention_found = any("hmm_attention_optimization" in opp.get("type", "") for opp in opportunities)
        self.assertTrue(hmm_attention_found)

    def test_identify_optimization_opportunities_no_issues(self):
        """Test optimization opportunities for well-configured model."""
        good_config = {
            "model_type": "rmm",
            "num_slots": 5,
            "slot_size": 32,
            "hidden_dim": 128,
            "num_objects": 3,
            "learning_rate": 0.001
        }

        opportunities = self.analyzer.identify_optimization_opportunities(good_config)

        # Should not identify major issues for well-configured model
        # (may still identify some minor opportunities)
        self.assertIsInstance(opportunities, list)

    def test_generate_model_report(self):
        """Test generating model report."""
        report = self.analyzer.generate_model_report(self.rmm_config)

        # Check that report contains expected sections
        self.assertIn("MODEL ANALYSIS REPORT", report)
        self.assertIn("Model Type: rmm", report)
        self.assertIn("Complexity Score:", report)
        self.assertIn("Parameter Estimate:", report)
        self.assertIn("Memory Estimate:", report)
        self.assertIn("Complexity Factors:", report)
        self.assertIn("Optimization Opportunities:", report)
        self.assertIn("Recommendations:", report)

    def test_generate_model_report_no_config(self):
        """Test generating report with no configuration."""
        report = self.analyzer.generate_model_report({})

        self.assertIn("MODEL ANALYSIS REPORT", report)
        self.assertIn("Unknown", report)  # Should handle unknown model type

    def test_analyze_base_model_types(self):
        """Test analysis of base model types (IMM, SMM, TMM)."""
        base_configs = [
            {
                "model_type": "imm",
                "hidden_dim": 256,
                "num_layers": 2,
                "learning_rate": 0.001
            },
            {
                "model_type": "smm",
                "num_slots": 8,
                "slot_dim": 64,
                "hidden_dim": 128,
                "learning_rate": 0.001
            },
            {
                "model_type": "tmm",
                "hidden_dim": 256,
                "num_layers": 2,
                "temporal_horizon": 10,
                "learning_rate": 0.001
            }
        ]

        for config in base_configs:
            analysis = self.analyzer.analyze_model_complexity(config)

            self.assertEqual(analysis["model_type"], config["model_type"])
            self.assertIn("complexity_score", analysis)
            self.assertIn("parameter_estimate", analysis)
            self.assertIn("memory_estimate_gb", analysis)
            self.assertIn("complexity_factors", analysis)

            # Check that complexity factors are present
            factors = analysis["complexity_factors"]
            self.assertIn("embedding_complexity", factors)
            self.assertIn("layer_complexity", factors)
            self.assertIn("vocabulary_size", factors)
            self.assertIn("model_depth", factors)


if __name__ == '__main__':
    unittest.main()
