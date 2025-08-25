"""
Tests for Architecture Visualizer

Comprehensive test suite for the AXIOM Architecture Visualizer
including visualization methods, analytics, and interactive features.
"""

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import shutil
import json
import sys

# Add src to path for imports
extra_dir = Path(__file__).parent.parent.parent.parent
src_dir = extra_dir / "src"
sys.path.insert(0, str(src_dir))

from model_setup_analysis.enhanced_architecture_visualizer import EnhancedAXIOMVisualizer


class TestEnhancedAXIOMVisualizer(unittest.TestCase):
    """Test cases for Enhanced AXIOM Visualizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

        # Mock the import calls to make matplotlib and networkx appear available
        import_mock = MagicMock()
        import_mock.pyplot = MagicMock()
        import_mock.patches = MagicMock()
        import_mock.gridspec = MagicMock()
        import_mock.nx = MagicMock()

        with patch('builtins.__import__', return_value=import_mock):
            self.visualizer = EnhancedAXIOMVisualizer(output_dir=self.temp_dir)

        # Override the availability flags for testing
        self.visualizer.matplotlib_available = True
        self.visualizer.networkx_available = True

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test visualizer initialization."""
        self.assertIsInstance(self.visualizer, EnhancedAXIOMVisualizer)
        self.assertTrue(self.visualizer.output_dir.exists())
        self.assertIsNotNone(self.visualizer.axiom_architecture)
        self.assertIsNotNone(self.visualizer.analytics_config)

    def test_architecture_data_loaded(self):
        """Test that architecture data is properly loaded."""
        arch = self.visualizer.axiom_architecture

        self.assertIn("models", arch)
        self.assertIn("relationships", arch)
        self.assertIn("data_flows", arch)

        # Check that all expected models are present
        expected_models = ["smm", "hsmm", "imm", "rmm", "tmm"]
        for model in expected_models:
            self.assertIn(model, arch["models"])
            self.assertIn("type", arch["models"][model])
            self.assertIn("components", arch["models"][model])
            self.assertIn("relationships", arch["models"][model])
            self.assertIn("parameters", arch["models"][model])
            self.assertIn("computation", arch["models"][model])

    @patch('matplotlib.pyplot.subplots')
    @patch('networkx.spring_layout')
    @patch('networkx.draw_networkx_nodes')
    @patch('networkx.draw_networkx_edges')
    @patch('networkx.draw_networkx_labels')
    @patch('networkx.draw_networkx_edge_labels')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_entity_relationship_model(self, mock_close, mock_savefig, mock_edge_labels,
                                             mock_labels, mock_edges, mock_nodes, mock_layout, mock_subplots):
        """Test entity-relationship model creation."""
        # Setup mocks
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_layout.return_value = {'smm': [0, 0], 'hsmm': [1, 1], 'imm': [2, 0]}

        # Test successful creation
        result = self.visualizer.create_entity_relationship_model("test_er.png")

        self.assertIsNotNone(result)
        self.assertTrue(Path(result).exists())

        # Verify matplotlib calls
        mock_subplots.assert_called_once()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_computation_graph_overview(self, mock_close, mock_savefig, mock_subplots):
        """Test computation graph overview creation."""
        # Setup mocks
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Test successful creation
        result = self.visualizer.create_computation_graph_overview("test_cg.png")

        self.assertIsNotNone(result)
        self.assertTrue(Path(result).exists())

        # Verify matplotlib calls
        mock_subplots.assert_called_once()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_architecture_analytics_dashboard(self, mock_close, mock_savefig, mock_figure):
        """Test architecture analytics dashboard creation."""
        # Setup mocks
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig

        # Test successful creation
        result = self.visualizer.create_architecture_analytics_dashboard("test_analytics.png")

        self.assertIsNotNone(result)
        self.assertTrue(Path(result).exists())

        # Verify matplotlib calls
        mock_figure.assert_called_once()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    def test_create_interactive_architecture_browser(self):
        """Test interactive architecture browser creation."""
        # Test successful creation
        result = self.visualizer.create_interactive_architecture_browser()

        self.assertTrue(result)

        # Check if interactive directory was created
        interactive_dir = self.temp_dir.parent / "interactive"
        self.assertTrue(interactive_dir.exists())

        # Check if HTML file was created
        html_file = interactive_dir / "architecture_browser.html"
        self.assertTrue(html_file.exists())

        # Check if data files were created
        data_files = ["model_data.json", "analytics_data.json", "metrics_data.json"]
        for data_file in data_files:
            self.assertTrue((interactive_dir / data_file).exists())

        # Check HTML content
        with open(html_file, 'r') as f:
            html_content = f.read()

        self.assertIn("AXIOM Architecture Interactive Browser", html_content)
        self.assertIn("Select Model to Explore", html_content)
        self.assertIn("Model Architecture", html_content)

    def test_export_architecture_report(self):
        """Test architecture report export."""
        # Test successful export
        result = self.visualizer.export_architecture_report("test_report.json")

        self.assertIsNotNone(result)
        self.assertTrue(Path(result).exists())

        # Check report content
        with open(result, 'r') as f:
            report = json.load(f)

        self.assertIn("architecture_overview", report)
        self.assertIn("analytics", report)
        self.assertIn("metrics", report)
        self.assertIn("recommendations", report)

        # Check metrics
        metrics = report["metrics"]
        self.assertIn("total_models", metrics)
        self.assertIn("total_relationships", metrics)
        self.assertIn("total_components", metrics)
        self.assertEqual(metrics["total_models"], 5)

    def test_no_matplotlib_fallback(self):
        """Test behavior when matplotlib is not available."""
        # Create visualizer with matplotlib disabled
        with patch('builtins.__import__', side_effect=ImportError("No module named 'matplotlib'")):
            visualizer = EnhancedAXIOMVisualizer(output_dir=self.temp_dir)

        self.assertFalse(visualizer.matplotlib_available)

        # Test that methods return None when matplotlib is not available
        result = visualizer.create_entity_relationship_model()
        self.assertIsNone(result)

        result = visualizer.create_computation_graph_overview()
        self.assertIsNone(result)

        result = visualizer.create_architecture_analytics_dashboard()
        self.assertIsNone(result)

    def test_no_networkx_fallback(self):
        """Test behavior when networkx is not available."""
        # Create visualizer with networkx disabled
        with patch('builtins.__import__', side_effect=ImportError("No module named 'networkx'")):
            visualizer = EnhancedAXIOMVisualizer(output_dir=self.temp_dir)

        self.assertFalse(visualizer.networkx_available)

        # Test that ER model returns None when networkx is not available
        result = visualizer.create_entity_relationship_model()
        self.assertIsNone(result)

    def test_analytics_integration(self):
        """Test analytics integration functionality."""
        # Check that analytics directory was created during initialization
        analytics_dir = self.temp_dir.parent / "analytics"
        self.assertTrue(analytics_dir.exists())

        # Check analytics config
        config = self.visualizer.analytics_config
        self.assertIn("metrics", config)
        self.assertIn("visualization_metrics", config)
        self.assertIn("model_complexity", config["metrics"])
        self.assertIn("entity_relationships", config["visualization_metrics"])

    def test_model_data_integrity(self):
        """Test that model data has correct structure and relationships."""
        models = self.visualizer.axiom_architecture["models"]

        # Test SMM model structure
        smm = models["smm"]
        self.assertEqual(smm["type"], "Slot Memory Model")
        self.assertIn("Input Layer", smm["components"])
        self.assertIn("Attention Mechanism", smm["components"])
        self.assertIn("input_dim", smm["parameters"])
        self.assertIn("linear_transforms", smm["computation"])

        # Test HSMM model structure
        hsmm = models["hsmm"]
        self.assertEqual(hsmm["type"], "Hierarchical Slot Memory Model")
        self.assertIn("Layer 0", hsmm["components"])
        self.assertIn("Hierarchical Integration", hsmm["components"])
        self.assertIn("num_layers", hsmm["parameters"])

        # Test IMM model structure
        imm = models["imm"]
        self.assertEqual(imm["type"], "Identity Memory Model")
        self.assertIn("Feature Extraction", imm["components"])
        self.assertIn("Identity Memory", imm["components"])

        # Test RMM model structure
        rmm = models["rmm"]
        self.assertEqual(rmm["type"], "Relational Memory Model")
        self.assertIn("State Processing", rmm["components"])
        self.assertIn("Dynamics Prediction", rmm["components"])

        # Test TMM model structure
        tmm = models["tmm"]
        self.assertEqual(tmm["type"], "Temporal Memory Model")
        self.assertIn("Current State", tmm["components"])
        self.assertIn("Temporal Prediction", tmm["components"])

    def test_relationship_data_integrity(self):
        """Test that relationship data has correct structure."""
        relationships = self.visualizer.axiom_architecture["relationships"]

        # Test that all expected relationships exist
        expected_relationships = [
            "smm_hsmm", "hsmm_imm", "hsmm_rmm", "rmm_tmm", "imm_rmm"
        ]

        for rel_key in expected_relationships:
            self.assertIn(rel_key, relationships)
            rel = relationships[rel_key]
            self.assertIn("type", rel)
            self.assertIn("description", rel)

        # Test relationship types
        self.assertEqual(relationships["smm_hsmm"]["type"], "hierarchical")
        self.assertEqual(relationships["hsmm_imm"]["type"], "identity")
        self.assertEqual(relationships["hsmm_rmm"]["type"], "relational")
        self.assertEqual(relationships["rmm_tmm"]["type"], "temporal")

    def test_data_flow_integrity(self):
        """Test that data flow data has correct structure."""
        data_flows = self.visualizer.axiom_architecture["data_flows"]

        # Test observation processing flow
        self.assertIn("observation_processing", data_flows)
        self.assertIn("SMM", data_flows["observation_processing"])
        self.assertIn("HSMM", data_flows["observation_processing"])

        # Test identity recognition flow
        self.assertIn("identity_recognition", data_flows)
        self.assertIn("IMM", data_flows["identity_recognition"])
        self.assertIn("RMM", data_flows["identity_recognition"])

        # Test relational dynamics flow
        self.assertIn("relational_dynamics", data_flows)
        self.assertIn("RMM", data_flows["relational_dynamics"])
        self.assertIn("TMM", data_flows["relational_dynamics"])

        # Test action prediction flow
        self.assertIn("action_prediction", data_flows)
        self.assertIn("RMM", data_flows["action_prediction"])
        self.assertIn("TMM", data_flows["action_prediction"])

    def test_computation_node_drawing(self):
        """Test computation node drawing functionality."""
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig, mock_ax = MagicMock(), MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            # Test with mock matplotlib available
            self.visualizer.matplotlib_available = True

            # Create a test stage
            stage_info = {
                'name': 'Test Stage',
                'position': (0.5, 0.5),
                'components': ['Comp1', 'Comp2'],
                'color': '#FF6B6B'
            }

            # This should not raise an exception
            self.visualizer._draw_computation_node(mock_ax, stage_info)

            # Verify that patches were added to the axis
            mock_ax.add_patch.assert_called()

    def test_data_flow_drawing(self):
        """Test data flow drawing functionality."""
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig, mock_ax = MagicMock(), MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            # Test with mock matplotlib available
            self.visualizer.matplotlib_available = True

            # Create test stages
            from_stage = {'position': (0.1, 0.8)}
            to_stage = {'position': (0.5, 0.6)}
            flow_type = 'Test Flow'

            # This should not raise an exception
            self.visualizer._draw_data_flow(mock_ax, from_stage, to_stage, flow_type)

            # Verify that patches were added to the axis
            mock_ax.add_patch.assert_called()

    def test_plot_model_complexity(self):
        """Test model complexity plotting."""
        with patch('matplotlib.pyplot.figure') as mock_figure:
            mock_fig = MagicMock()
            mock_figure.return_value = mock_fig

            self.visualizer.matplotlib_available = True

            # This should not raise an exception
            self.visualizer._plot_model_complexity(mock_fig.add_subplot())

    def test_plot_component_relationships(self):
        """Test component relationships plotting."""
        with patch('matplotlib.pyplot.figure') as mock_figure:
            mock_fig = MagicMock()
            mock_figure.return_value = mock_fig

            self.visualizer.matplotlib_available = True

            # This should not raise an exception
            self.visualizer._plot_component_relationships(mock_fig.add_subplot())

    def test_plot_data_flow_analysis(self):
        """Test data flow analysis plotting."""
        with patch('matplotlib.pyplot.figure') as mock_figure:
            mock_fig = MagicMock()
            mock_figure.return_value = mock_fig

            self.visualizer.matplotlib_available = True

            # This should not raise an exception
            self.visualizer._plot_data_flow_analysis(mock_fig.add_subplot())

    def test_plot_performance_metrics(self):
        """Test performance metrics plotting."""
        with patch('matplotlib.pyplot.figure') as mock_figure:
            mock_fig = MagicMock()
            mock_figure.return_value = mock_fig

            self.visualizer.matplotlib_available = True

            # This should not raise an exception
            self.visualizer._plot_performance_metrics(mock_fig.add_subplot())

    def test_plot_architecture_patterns(self):
        """Test architecture patterns plotting."""
        with patch('matplotlib.pyplot.figure') as mock_figure:
            mock_fig = MagicMock()
            mock_figure = mock_fig

            self.visualizer.matplotlib_available = True

            # This should not raise an exception
            self.visualizer._plot_architecture_patterns(mock_fig.add_subplot())

    def test_plot_computation_graph_metrics(self):
        """Test computation graph metrics plotting."""
        with patch('matplotlib.pyplot.figure') as mock_figure:
            mock_fig = MagicMock()
            mock_figure.return_value = mock_fig

            self.visualizer.matplotlib_available = True

            # This should not raise an exception
            self.visualizer._plot_computation_graph_metrics(mock_fig.add_subplot())

    def test_html_generation(self):
        """Test HTML generation for interactive browser."""
        html_content = self.visualizer._generate_interactive_html()

        self.assertIn("AXIOM Architecture Interactive Browser", html_content)
        self.assertIn("Select Model to Explore", html_content)
        self.assertIn("Model Architecture", html_content)
        self.assertIn("Performance Metrics", html_content)
        self.assertIn("Component Relationships", html_content)

    def test_interactive_data_files_creation(self):
        """Test creation of interactive data files."""
        interactive_dir = self.temp_dir / "test_interactive"
        interactive_dir.mkdir()

        self.visualizer._create_interactive_data_files(interactive_dir)

        # Check if all expected files were created
        expected_files = ["model_data.json", "analytics_data.json", "metrics_data.json"]
        for filename in expected_files:
            file_path = interactive_dir / filename
            self.assertTrue(file_path.exists())

            # Check that files contain valid JSON
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.assertIsInstance(data, dict)


if __name__ == '__main__':
    unittest.main()
