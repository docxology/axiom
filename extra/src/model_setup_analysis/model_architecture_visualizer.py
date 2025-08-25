"""
Model Architecture Visualizer for AXIOM Extensions

Provides comprehensive visualization methods for AXIOM model architectures,
state spaces, and component relationships.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class AXIOMModelVisualizer:
    """
    Comprehensive visualizer for AXIOM model architectures and state spaces.

    This class provides methods to:
    - Visualize individual model architectures (SMM, HSMM, IMM, RMM, TMM)
    - Generate component relationship diagrams
    - Create state space representations
    - Produce overall architecture summaries
    - Generate modular, reusable visualization components
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the model architecture visualizer.

        Args:
            output_dir: Directory for saving visualizations (if None, uses output config)
        """
        # Import output configuration
        try:
            from ..output_config import output_config
            if output_dir is None:
                self.output_dir = output_config.get_path_for("visualizations", "static")
            else:
                self.output_dir = Path(output_dir)
        except ImportError:
            self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "model_visualizations"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Initialize visualization components
        self._setup_visualization_components()

    def _setup_visualization_components(self):
        """Set up modular visualization components."""
        self.components = {
            'architecture': ArchitectureVisualizer(self.output_dir),
            'state_space': StateSpaceVisualizer(self.output_dir),
            'relationships': RelationshipVisualizer(self.output_dir),
            'comprehensive': ComprehensiveVisualizer(self.output_dir)
        }

    def visualize_smm_architecture(self, smm_config: Dict[str, Any],
                                 filename: Optional[str] = None) -> Optional[str]:
        """
        Visualize SMM (Slot Memory Model) architecture.

        Args:
            smm_config: SMM configuration dictionary
            filename: Optional output filename

        Returns:
            Path to the generated visualization file
        """
        return self.components['architecture'].visualize_smm(
            smm_config, filename or "smm_architecture.png"
        )

    def visualize_hsmm_architecture(self, hsmm_config: List[Dict[str, Any]],
                                   filename: Optional[str] = None) -> Optional[str]:
        """
        Visualize HSMM (Hierarchical Slot Memory Model) architecture.

        Args:
            hsmm_config: List of SMM configurations for each layer
            filename: Optional output filename

        Returns:
            Path to the generated visualization file
        """
        return self.components['architecture'].visualize_hsmm(
            hsmm_config, filename or "hsmm_architecture.png"
        )

    def visualize_imm_architecture(self, imm_config: Dict[str, Any],
                                 filename: Optional[str] = None) -> Optional[str]:
        """
        Visualize IMM (Identity Memory Model) architecture.

        Args:
            imm_config: IMM configuration dictionary
            filename: Optional output filename

        Returns:
            Path to the generated visualization file
        """
        return self.components['architecture'].visualize_imm(
            imm_config, filename or "imm_architecture.png"
        )

    def visualize_rmm_architecture(self, rmm_config: Dict[str, Any],
                                 filename: Optional[str] = None) -> Optional[str]:
        """
        Visualize RMM (Relational Memory Model) architecture.

        Args:
            rmm_config: RMM configuration dictionary
            filename: Optional output filename

        Returns:
            Path to the generated visualization file
        """
        return self.components['architecture'].visualize_rmm(
            rmm_config, filename or "rmm_architecture.png"
        )

    def visualize_tmm_architecture(self, tmm_config: Dict[str, Any],
                                 filename: Optional[str] = None) -> Optional[str]:
        """
        Visualize TMM (Temporal Memory Model) architecture.

        Args:
            tmm_config: TMM configuration dictionary
            filename: Optional output filename

        Returns:
            Path to the generated visualization file
        """
        return self.components['architecture'].visualize_tmm(
            tmm_config, filename or "tmm_architecture.png"
        )

    def visualize_hybrid_utils_architecture(self, filename: Optional[str] = None) -> Optional[str]:
        """
        Visualize hybrid model utilities architecture.

        Args:
            filename: Optional output filename

        Returns:
            Path to the generated visualization file
        """
        return self.components['architecture'].visualize_hybrid_utils(
            filename or "hybrid_utils_architecture.png"
        )

    def visualize_all_model_architectures(self, config_dict: Dict[str, Any],
                                        filename: Optional[str] = None) -> Optional[str]:
        """
        Visualize all model architectures in a single comprehensive diagram.

        Args:
            config_dict: Dictionary containing configurations for all models
            filename: Optional output filename

        Returns:
            Path to the generated visualization file
        """
        return self.components['comprehensive'].visualize_all_architectures(
            config_dict, filename or "all_model_architectures.png"
        )

    def create_graphical_abstract(self, config_dict: Dict[str, Any],
                                filename: Optional[str] = None) -> Optional[str]:
        """
        Create a comprehensive graphical abstract of the entire AXIOM architecture.

        Args:
            config_dict: Dictionary containing configurations for all models
            filename: Optional output filename

        Returns:
            Path to the generated graphical abstract file
        """
        return self.components['comprehensive'].create_graphical_abstract(
            config_dict, filename or "axiom_graphical_abstract.png"
        )

    def visualize_model_relationships(self, filename: Optional[str] = None) -> Optional[str]:
        """
        Visualize relationships between different AXIOM model components.

        Args:
            filename: Optional output filename

        Returns:
            Path to the generated visualization file
        """
        return self.components['relationships'].visualize_model_relationships(
            filename or "model_relationships.png"
        )

    def visualize_state_space_evolution(self, model_type: str, config: Dict[str, Any],
                                      evolution_data: List[Dict[str, Any]],
                                      filename: Optional[str] = None) -> Optional[str]:
        """
        Visualize how state spaces evolve over time for a given model.

        Args:
            model_type: Type of model ('smm', 'hsmm', 'imm', 'rmm', 'tmm')
            config: Model configuration
            evolution_data: Time series of state space data
            filename: Optional output filename

        Returns:
            Path to the generated visualization file
        """
        return self.components['state_space'].visualize_state_evolution(
            model_type, config, evolution_data,
            filename or f"{model_type}_state_evolution.png"
        )


class ArchitectureVisualizer:
    """Handles visualization of individual model architectures."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self._setup_matplotlib()

    def _setup_matplotlib(self):
        """Set up matplotlib for architecture visualization."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib.patches import FancyBboxPatch, ConnectionPatch
            self.plt = plt
            self.patches = patches
            self.FancyBboxPatch = FancyBboxPatch
            self.ConnectionPatch = ConnectionPatch
            self.matplotlib_available = True
        except ImportError:
            self.matplotlib_available = False
            self.logger = logging.getLogger(__name__)
            self.logger.warning("Matplotlib not available for architecture visualization")

    def visualize_smm(self, config: Dict[str, Any], filename: str) -> Optional[str]:
        """Visualize SMM architecture."""
        if not self.matplotlib_available:
            return None

        fig, ax = self.plt.subplots(figsize=(12, 8))
        ax.set_title("Slot Memory Model (SMM) Architecture", fontsize=16, fontweight='bold')

        # Extract configuration
        width = config.get('width', 160)
        height = config.get('height', 210)
        input_dim = config.get('input_dim', 5)
        slot_dim = config.get('slot_dim', 2)
        num_slots = config.get('num_slots', 32)

        # Draw input layer
        self._draw_layer_box(ax, (0.1, 0.8), (0.15, 0.1), f"Input\n({input_dim}D)", 'lightblue')

        # Draw attention mechanism
        self._draw_layer_box(ax, (0.3, 0.8), (0.2, 0.1), f"Attention\n{num_slots} slots", 'lightgreen')

        # Draw slot representations
        for i in range(min(num_slots, 8)):  # Show first 8 slots
            y_pos = 0.6 - (i * 0.05)
            color = self.plt.cm.viridis(i / num_slots)
            self._draw_layer_box(ax, (0.55, y_pos), (0.15, 0.03),
                               f"Slot {i}\n({slot_dim}D)", color)

        # Draw output layer
        self._draw_layer_box(ax, (0.8, 0.5), (0.15, 0.1),
                           f"Output\n({height}×{width})", 'lightcoral')

        # Draw connections
        self._draw_connections(ax, [(0.25, 0.85), (0.3, 0.85)],
                             [(0.5, 0.85), (0.55, 0.85)])
        self._draw_connections(ax, [(0.5, 0.85), (0.55, 0.85)],
                             [(0.7, 0.6), (0.8, 0.6)])

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        self.plt.close(fig)

        return str(filepath)

    def visualize_hsmm(self, layer_configs: List[Dict[str, Any]], filename: str) -> Optional[str]:
        """Visualize HSMM architecture."""
        if not self.matplotlib_available:
            return None

        fig, ax = self.plt.subplots(figsize=(16, 10))
        ax.set_title("Hierarchical Slot Memory Model (HSMM) Architecture",
                    fontsize=16, fontweight='bold')

        num_layers = len(layer_configs)
        layer_width = 0.8 / num_layers

        # Draw hierarchical layers
        for i, config in enumerate(layer_configs):
            x_pos = 0.1 + (i * layer_width)

            # Layer box
            self._draw_layer_box(ax, (x_pos, 0.8), (layer_width*0.8, 0.1),
                               f"Layer {i}\nSMM", 'lightblue')

            # Slot representations
            num_slots = config.get('num_slots', 8)
            slot_height = 0.4 / num_slots

            for j in range(min(num_slots, 4)):  # Show first 4 slots
                y_pos = 0.6 - (j * slot_height)
                color = self.plt.cm.plasma(j / num_slots)
                self._draw_layer_box(ax, (x_pos, y_pos), (layer_width*0.6, slot_height*0.8),
                                   f"Slot {j}", color)

            # Draw connections between layers
            if i < num_layers - 1:
                next_x = 0.1 + ((i + 1) * layer_width)
                self._draw_connections(ax, [(x_pos + layer_width*0.8, 0.5)],
                                     [(next_x, 0.5)])

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        self.plt.close(fig)

        return str(filepath)

    def visualize_imm(self, config: Dict[str, Any], filename: str) -> Optional[str]:
        """Visualize IMM architecture."""
        if not self.matplotlib_available:
            return None

        fig, ax = self.plt.subplots(figsize=(12, 8))
        ax.set_title("Identity Memory Model (IMM) Architecture", fontsize=16, fontweight='bold')

        # Extract configuration
        num_object_types = config.get('num_object_types', 32)
        num_features = config.get('num_features', 5)

        # Draw input layer
        self._draw_layer_box(ax, (0.1, 0.8), (0.15, 0.1), f"Input\n({num_features}D)", 'lightblue')

        # Draw feature processing
        self._draw_layer_box(ax, (0.3, 0.8), (0.2, 0.1), "Feature\nProcessing", 'lightgreen')

        # Draw identity clusters
        cluster_height = 0.4 / min(num_object_types, 8)
        for i in range(min(num_object_types, 8)):
            y_pos = 0.6 - (i * cluster_height)
            color = self.plt.cm.coolwarm(i / num_object_types)
            self._draw_layer_box(ax, (0.55, y_pos), (0.15, cluster_height*0.8),
                               f"Identity {i}", color)

        # Draw output layer
        self._draw_layer_box(ax, (0.8, 0.5), (0.15, 0.1),
                           f"Output\nIdentity", 'lightcoral')

        # Draw connections
        self._draw_connections(ax, [(0.25, 0.85), (0.3, 0.85)],
                             [(0.5, 0.85), (0.55, 0.85)])
        self._draw_connections(ax, [(0.5, 0.85), (0.55, 0.85)],
                             [(0.7, 0.6), (0.8, 0.6)])

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        self.plt.close(fig)

        return str(filepath)

    def visualize_rmm(self, config: Dict[str, Any], filename: str) -> Optional[str]:
        """Visualize RMM architecture."""
        if not self.matplotlib_available:
            return None

        fig, ax = self.plt.subplots(figsize=(14, 10))
        ax.set_title("Relational Memory Model (RMM) Architecture", fontsize=16, fontweight='bold')

        # Extract configuration
        num_components_per_switch = config.get('num_components_per_switch', 25)
        num_switches = config.get('num_switches', 100)
        num_object_types = config.get('num_object_types', 32)

        # Draw input components
        self._draw_layer_box(ax, (0.05, 0.8), (0.15, 0.1), "State\nInput", 'lightblue')
        self._draw_layer_box(ax, (0.05, 0.6), (0.15, 0.1), "Action\nInput", 'lightgreen')
        self._draw_layer_box(ax, (0.05, 0.4), (0.15, 0.1), "Reward\nInput", 'lightyellow')

        # Draw TMM switch prediction
        self._draw_layer_box(ax, (0.25, 0.7), (0.2, 0.1),
                           f"TMM Switch\n({num_switches} types)", 'orange')

        # Draw relational components
        self._draw_layer_box(ax, (0.5, 0.8), (0.2, 0.1),
                           f"Self Identity\n({num_object_types})", 'lightcoral')
        self._draw_layer_box(ax, (0.5, 0.6), (0.2, 0.1),
                           f"Other Identity\n({num_object_types})", 'lightcoral')
        self._draw_layer_box(ax, (0.5, 0.4), (0.2, 0.1), "Interactions", 'lightpink')

        # Draw mixture components
        total_components = num_components_per_switch * num_switches
        self._draw_layer_box(ax, (0.75, 0.6), (0.2, 0.1),
                           f"Mixture\n({total_components} components)", 'purple')

        # Draw output predictions
        self._draw_layer_box(ax, (0.9, 0.7), (0.08, 0.08), "TMM\nSwitch", 'red')
        self._draw_layer_box(ax, (0.9, 0.5), (0.08, 0.08), "Reward", 'blue')

        # Draw connections
        self._draw_connections(ax, [(0.2, 0.75)], [(0.25, 0.75)])
        self._draw_connections(ax, [(0.2, 0.65), (0.2, 0.45)], [(0.5, 0.75), (0.5, 0.65)])
        self._draw_connections(ax, [(0.7, 0.75), (0.7, 0.65)], [(0.75, 0.65)])
        self._draw_connections(ax, [(0.85, 0.65)], [(0.9, 0.65), (0.9, 0.55)])

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        self.plt.close(fig)

        return str(filepath)

    def visualize_tmm(self, config: Dict[str, Any], filename: str) -> Optional[str]:
        """Visualize TMM architecture."""
        if not self.matplotlib_available:
            return None

        fig, ax = self.plt.subplots(figsize=(12, 8))
        ax.set_title("Temporal Memory Model (TMM) Architecture", fontsize=16, fontweight='bold')

        # Extract configuration
        n_total_components = config.get('n_total_components', 200)
        state_dim = config.get('state_dim', 2)
        use_velocity = config.get('use_velocity', True)

        # Draw input state
        self._draw_layer_box(ax, (0.1, 0.8), (0.15, 0.1),
                           f"Current State\n({state_dim}D)", 'lightblue')

        # Draw transition components
        component_height = 0.4 / min(n_total_components, 6)
        for i in range(min(n_total_components, 6)):
            y_pos = 0.6 - (i * component_height)
            color = self.plt.cm.tab10(i % 10)
            self._draw_layer_box(ax, (0.35, y_pos), (0.2, component_height*0.8),
                               f"Dynamics {i}", color)

        # Draw prediction
        self._draw_layer_box(ax, (0.65, 0.6), (0.2, 0.1),
                           f"Next State\n({state_dim}D)", 'lightgreen')

        # Draw velocity component if applicable
        if use_velocity:
            self._draw_layer_box(ax, (0.65, 0.4), (0.2, 0.1),
                               "Velocity\nIntegration", 'orange')

        # Draw connections
        self._draw_connections(ax, [(0.25, 0.85)], [(0.35, 0.85)])
        self._draw_connections(ax, [(0.55, 0.65)], [(0.65, 0.65)])

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        self.plt.close(fig)

        return str(filepath)

    def visualize_hybrid_utils(self, filename: str) -> Optional[str]:
        """Visualize hybrid model utilities architecture."""
        if not self.matplotlib_available:
            return None

        fig, ax = self.plt.subplots(figsize=(14, 10))
        ax.set_title("Hybrid Model Utilities Architecture", fontsize=16, fontweight='bold')

        # Draw utility functions
        utils = [
            "create_mm", "train_step_fn", "create_mvn",
            "block_diag", "velocity_augmentation", "state_prediction"
        ]

        for i, util in enumerate(utils):
            x_pos = 0.1 + (i % 3) * 0.3
            y_pos = 0.8 - (i // 3) * 0.2
            color = self.plt.cm.Set3(i % 12)
            self._draw_layer_box(ax, (x_pos, y_pos), (0.25, 0.1), util, color)

        # Draw connections showing utility relationships
        self._draw_connections(ax, [(0.35, 0.85)], [(0.4, 0.85)])
        self._draw_connections(ax, [(0.65, 0.85)], [(0.7, 0.85)])
        self._draw_connections(ax, [(0.35, 0.65)], [(0.4, 0.65)])

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        self.plt.close(fig)

        return str(filepath)

    def _draw_layer_box(self, ax, position, size, text, color):
        """Draw a layer box with text."""
        x, y = position
        width, height = size

        # Create rounded rectangle
        box = self.FancyBboxPatch((x, y), width, height,
                                boxstyle="round,pad=0.02",
                                facecolor=color, alpha=0.8)
        ax.add_patch(box)

        # Add text
        ax.text(x + width/2, y + height/2, text,
               ha='center', va='center', fontsize=10, fontweight='bold')

    def _draw_connections(self, ax, start_points, end_points):
        """Draw connections between layers."""
        for start, end in zip(start_points, end_points):
            conn = self.ConnectionPatch(start, end, "data", "data",
                                      arrowstyle="->", shrinkA=5, shrinkB=5,
                                      mutation_scale=20, fc="k")
            ax.add_patch(conn)


class StateSpaceVisualizer:
    """Handles visualization of state spaces and their evolution."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self._setup_matplotlib()

    def _setup_matplotlib(self):
        """Set up matplotlib for state space visualization."""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            self.plt = plt
            self.Axes3D = Axes3D
            self.matplotlib_available = True
        except ImportError:
            self.matplotlib_available = False

    def visualize_state_evolution(self, model_type: str, config: Dict[str, Any],
                                evolution_data: List[Dict[str, Any]], filename: str) -> Optional[str]:
        """Visualize state space evolution over time."""
        if not self.matplotlib_available:
            return None

        fig, ax = self.plt.subplots(figsize=(12, 8))

        # Plot state evolution based on model type
        if model_type == 'smm':
            ax.set_title("SMM State Space Evolution")
            self._plot_smm_evolution(ax, config, evolution_data)
        elif model_type == 'hsmm':
            ax.set_title("HSMM Hierarchical State Space Evolution")
            self._plot_hsmm_evolution(ax, config, evolution_data)
        elif model_type == 'rmm':
            ax.set_title("RMM Relational State Space Evolution")
            self._plot_rmm_evolution(ax, config, evolution_data)
        elif model_type == 'tmm':
            ax.set_title("TMM Temporal State Space Evolution")
            self._plot_tmm_evolution(ax, config, evolution_data)

        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        self.plt.close(fig)

        return str(filepath)

    def _plot_smm_evolution(self, ax, config, evolution_data):
        """Plot SMM state evolution."""
        for i, data in enumerate(evolution_data):
            slots = data.get('slots', [])
            for j, slot in enumerate(slots):
                ax.scatter(slot[0], slot[1], c=f'C{j}', alpha=0.6, s=50)
        ax.set_xlabel('Position X')
        ax.set_ylabel('Position Y')

    def _plot_hsmm_evolution(self, ax, config, evolution_data):
        """Plot HSMM state evolution."""
        for i, data in enumerate(evolution_data):
            for layer_idx, layer_data in enumerate(data.get('layers', [])):
                ax.scatter(layer_data['x'], layer_data['y'],
                          c=f'C{layer_idx}', alpha=0.6, s=30)
        ax.set_xlabel('Hierarchical Position X')
        ax.set_ylabel('Hierarchical Position Y')

    def _plot_rmm_evolution(self, ax, config, evolution_data):
        """Plot RMM state evolution."""
        for i, data in enumerate(evolution_data):
            relations = data.get('relations', [])
            for rel in relations:
                ax.arrow(rel['from'][0], rel['from'][1],
                        rel['to'][0] - rel['from'][0],
                        rel['to'][1] - rel['from'][1],
                        head_width=0.1, alpha=0.6)
        ax.set_xlabel('Relational Position X')
        ax.set_ylabel('Relational Position Y')

    def _plot_tmm_evolution(self, ax, config, evolution_data):
        """Plot TMM state evolution."""
        for i, data in enumerate(evolution_data):
            trajectory = data.get('trajectory', [])
            if trajectory:
                xs, ys = zip(*trajectory)
                ax.plot(xs, ys, 'o-', alpha=0.7, markersize=4)
        ax.set_xlabel('Temporal Position X')
        ax.set_ylabel('Temporal Position Y')


class RelationshipVisualizer:
    """Handles visualization of relationships between model components."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self._setup_matplotlib()

    def _setup_matplotlib(self):
        """Set up matplotlib for relationship visualization."""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            self.plt = plt
            self.nx = nx
            self.matplotlib_available = True
        except ImportError:
            self.matplotlib_available = False

    def visualize_model_relationships(self, filename: str) -> Optional[str]:
        """Visualize relationships between AXIOM model components."""
        if not self.matplotlib_available:
            return None

        fig, ax = self.plt.subplots(figsize=(14, 10))

        # Create relationship graph
        G = self.nx.DiGraph()

        # Add nodes
        models = ['SMM', 'HSMM', 'IMM', 'RMM', 'TMM', 'Utils']
        for model in models:
            G.add_node(model, type='model')

        # Add relationships
        relationships = [
            ('SMM', 'HSMM', 'hierarchical'),
            ('HSMM', 'IMM', 'identity'),
            ('HSMM', 'RMM', 'relational'),
            ('RMM', 'TMM', 'temporal'),
            ('IMM', 'RMM', 'identity'),
            ('Utils', 'SMM', 'utilities'),
            ('Utils', 'HSMM', 'utilities'),
            ('Utils', 'IMM', 'utilities'),
            ('Utils', 'RMM', 'utilities'),
            ('Utils', 'TMM', 'utilities'),
        ]

        for rel in relationships:
            G.add_edge(rel[0], rel[1], relation=rel[2])

        # Draw graph
        pos = self.nx.spring_layout(G, k=1, iterations=50)

        # Draw nodes
        node_colors = ['lightblue', 'lightgreen', 'lightcoral', 'orange', 'purple', 'gray']
        self.nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                                   node_size=3000, alpha=0.8)

        # Draw edges
        self.nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray',
                                   arrows=True, arrowsize=20)

        # Draw labels
        self.nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_weight='bold')

        # Draw edge labels
        edge_labels = {(u, v): d['relation'] for u, v, d in G.edges(data=True)}
        self.nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels,
                                         font_size=10)

        ax.set_title("AXIOM Model Component Relationships", fontsize=16, fontweight='bold')
        ax.axis('off')

        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        self.plt.close(fig)

        return str(filepath)


class ComprehensiveVisualizer:
    """Handles comprehensive visualizations of the entire AXIOM architecture."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self._setup_matplotlib()

    def _setup_matplotlib(self):
        """Set up matplotlib for comprehensive visualization."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            self.plt = plt
            self.gridspec = gridspec
            self.matplotlib_available = True
        except ImportError:
            self.matplotlib_available = False

    def visualize_all_architectures(self, config_dict: Dict[str, Any], filename: str) -> Optional[str]:
        """Visualize all model architectures in a single comprehensive diagram."""
        if not self.matplotlib_available:
            return None

        fig = self.plt.figure(figsize=(20, 16))
        gs = self.gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.2)

        # Create subplots for each model
        axes = []
        for i in range(6):
            row = i // 2
            col = i % 2
            axes.append(fig.add_subplot(gs[row, col]))

        # Visualize each model
        model_names = ['SMM', 'HSMM', 'IMM', 'RMM', 'TMM', 'Utils']

        for i, (ax, model_name) in enumerate(zip(axes, model_names)):
            ax.set_title(f"{model_name} Architecture", fontsize=14, fontweight='bold')

            if model_name == 'SMM':
                self._draw_simple_architecture(ax, ['Input', 'Attention', 'Slots', 'Output'],
                                              'lightblue')
            elif model_name == 'HSMM':
                self._draw_simple_architecture(ax, ['Layer 0', 'Layer 1', 'Layer 2', 'Hierarchical'],
                                              'lightgreen')
            elif model_name == 'IMM':
                self._draw_simple_architecture(ax, ['Features', 'Identity', 'Clusters', 'Output'],
                                              'lightcoral')
            elif model_name == 'RMM':
                self._draw_simple_architecture(ax, ['State', 'Relations', 'Mixture', 'Predictions'],
                                              'orange')
            elif model_name == 'TMM':
                self._draw_simple_architecture(ax, ['Current', 'Dynamics', 'Prediction', 'Temporal'],
                                              'purple')
            elif model_name == 'Utils':
                self._draw_simple_architecture(ax, ['create_mm', 'train_step', 'utilities', 'helpers'],
                                              'gray')

        fig.suptitle("AXIOM Model Architectures Overview", fontsize=18, fontweight='bold')

        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        self.plt.close(fig)

        return str(filepath)

    def create_graphical_abstract(self, config_dict: Dict[str, Any], filename: str) -> Optional[str]:
        """Create a comprehensive graphical abstract of the entire AXIOM architecture."""
        if not self.matplotlib_available:
            return None

        fig, ax = self.plt.subplots(figsize=(16, 12))

        # Main title
        ax.text(0.5, 0.95, "AXIOM: Learning to Play Games in Minutes",
               ha='center', va='center', fontsize=18, fontweight='bold',
               transform=ax.transAxes)

        # Draw overall architecture flow
        self._draw_overall_architecture(ax)

        # Add model descriptions
        descriptions = {
            'smm': "Slot Memory Model\n(Object-centric representation)",
            'hsmm': "Hierarchical SMM\n(Multi-layer object hierarchy)",
            'imm': "Identity Memory Model\n(Object identity recognition)",
            'rmm': "Relational Memory Model\n(Object interactions & dynamics)",
            'tmm': "Temporal Memory Model\n(Temporal state prediction)"
        }

        y_positions = [0.75, 0.6, 0.45, 0.3, 0.15]
        for i, (model, desc) in enumerate(descriptions.items()):
            ax.text(0.02, y_positions[i], f"{model.upper()}:",
                   ha='left', va='center', fontsize=12, fontweight='bold')
            ax.text(0.15, y_positions[i], desc,
                   ha='left', va='center', fontsize=10)

        # Add data flow arrows
        self._draw_data_flow(ax)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        self.plt.close(fig)

        return str(filepath)

    def _draw_simple_architecture(self, ax, components, color):
        """Draw a simple architecture diagram."""
        x_positions = [0.2 + i * 0.2 for i in range(len(components))]

        for i, (x, component) in enumerate(zip(x_positions, components)):
            # Draw component box
            rect = self.plt.Rectangle((x-0.08, 0.4), 0.16, 0.2,
                                    facecolor=color, alpha=0.8, edgecolor='black')
            ax.add_patch(rect)

            # Add text
            ax.text(x, 0.5, component, ha='center', va='center',
                   fontsize=10, fontweight='bold')

            # Draw connections
            if i < len(components) - 1:
                ax.arrow(x + 0.08, 0.5, 0.12, 0, head_width=0.02,
                        head_length=0.02, fc='black', ec='black')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    def _draw_overall_architecture(self, ax):
        """Draw the overall AXIOM architecture flow."""
        # Main components
        components = [
            (0.2, 0.8, "Observation\nProcessing", 'lightblue'),
            (0.4, 0.8, "Hierarchical\nSMM", 'lightgreen'),
            (0.6, 0.8, "Identity\nRecognition", 'lightcoral'),
            (0.8, 0.8, "Relational\nDynamics", 'orange'),
            (0.9, 0.6, "Action\nPrediction", 'purple')
        ]

        for x, y, label, color in components:
            # Draw component box
            rect = self.plt.Rectangle((x-0.08, y-0.05), 0.16, 0.1,
                                    facecolor=color, alpha=0.8, edgecolor='black')
            ax.add_patch(rect)

            # Add text
            ax.text(x, y, label, ha='center', va='center',
                   fontsize=10, fontweight='bold')

        # Draw flow arrows
        arrow_positions = [(0.28, 0.8), (0.48, 0.8), (0.68, 0.8), (0.85, 0.75)]
        for i, (x, y) in enumerate(arrow_positions[:-1]):
            next_x, next_y = arrow_positions[i + 1]
            ax.arrow(x, y, next_x - x, next_y - y, head_width=0.02,
                    head_length=0.02, fc='black', ec='black')

    def _draw_data_flow(self, ax):
        """Draw data flow arrows."""
        # Input flow
        ax.arrow(0.05, 0.85, 0.12, 0, head_width=0.02, head_length=0.02,
                fc='blue', ec='blue', alpha=0.7)
        ax.text(0.1, 0.87, "Game State", fontsize=10)

        # Processing flow
        ax.arrow(0.32, 0.8, 0.06, 0, head_width=0.02, head_length=0.02,
                fc='green', ec='green', alpha=0.7)

        # Output flow
        ax.arrow(0.88, 0.55, 0.08, -0.15, head_width=0.02, head_length=0.02,
                fc='red', ec='red', alpha=0.7)
        ax.text(0.9, 0.45, "Game Action", fontsize=10, ha='center')

