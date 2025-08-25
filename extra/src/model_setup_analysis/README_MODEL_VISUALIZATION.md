# AXIOM Model Architecture Visualizer

A comprehensive visualization system for generating programmatic images of AXIOM model architectures, state spaces, and component relationships.

## Overview

The AXIOM Model Architecture Visualizer provides modular, reusable methods for creating detailed visualizations of all AXIOM model components:

- **SMM (Slot Memory Model)**: Object-centric representation learning
- **HSMM (Hierarchical SMM)**: Multi-layer object hierarchy processing
- **IMM (Identity Memory Model)**: Object identity recognition and clustering
- **RMM (Relational Memory Model)**: Object interactions and dynamics prediction
- **TMM (Temporal Memory Model)**: Temporal state prediction and trajectory modeling
- **Hybrid Utils**: Utility functions for hybrid model architectures

## Features

### 🏗️ **Architecture Visualization**
- Individual model architecture diagrams
- Component flow and data pathways
- Parameter and configuration visualization
- Modular design for easy extension

### 📊 **State Space Visualization**
- State space evolution over time
- Multi-dimensional representation
- Temporal trajectory visualization
- Hierarchical state progression

### 🔗 **Relationship Visualization**
- Inter-model component relationships
- Data flow dependencies
- Hierarchical connections
- System integration overview

### 🎨 **Comprehensive Graphics**
- Overall system architecture abstract
- Multi-model comparison views
- Interactive and static export formats
- High-resolution publication-quality outputs

## Installation Requirements

```bash
# Core dependencies
pip install matplotlib numpy

# Optional for advanced graph visualization
pip install networkx

# For high-quality output
pip install pillow
```

## Quick Start

### Basic Usage

```python
from model_setup_analysis.model_architecture_visualizer import AXIOMModelVisualizer

# Initialize visualizer
visualizer = AXIOMModelVisualizer()

# Example configurations
smm_config = {
    'width': 160, 'height': 210, 'input_dim': 5,
    'slot_dim': 2, 'num_slots': 32
}

# Generate SMM architecture visualization
smm_path = visualizer.visualize_smm_architecture(
    smm_config, "smm_architecture.png"
)
print(f"SMM architecture saved to: {smm_path}")
```

### Running the Complete Demo

```bash
cd /path/to/axiom/extra/src/model_setup_analysis
python model_visualization_demo.py
```

This will generate:
- 6 individual model architecture diagrams
- 4 state space evolution visualizations
- 1 model relationship diagram
- 1 comprehensive architecture overview
- 1 graphical abstract
- 1 JSON summary file

## API Reference

### AXIOMModelVisualizer

Main visualization interface with methods for all model types.

#### Individual Model Architectures

```python
# SMM Architecture
smm_path = visualizer.visualize_smm_architecture(smm_config, filename)

# HSMM Architecture (hierarchical)
hsmm_path = visualizer.visualize_hsmm_architecture(layer_configs, filename)

# IMM Architecture
imm_path = visualizer.visualize_imm_architecture(imm_config, filename)

# RMM Architecture
rmm_path = visualizer.visualize_rmm_architecture(rmm_config, filename)

# TMM Architecture
tmm_path = visualizer.visualize_tmm_architecture(tmm_config, filename)

# Hybrid Utils Architecture
utils_path = visualizer.visualize_hybrid_utils_architecture(filename)
```

#### State Space Evolution

```python
# Visualize state evolution for any model type
evolution_path = visualizer.visualize_state_space_evolution(
    model_type,      # 'smm', 'hsmm', 'imm', 'rmm', 'tmm'
    config,          # Model configuration
    evolution_data,  # Time series data
    filename
)
```

#### Relationships and Overviews

```python
# Model relationships
relationships_path = visualizer.visualize_model_relationships(filename)

# All architectures overview
all_architectures_path = visualizer.visualize_all_model_architectures(
    config_dict, filename
)

# Graphical abstract
abstract_path = visualizer.create_graphical_abstract(
    config_dict, filename
)
```

## Model Configuration Examples

### SMM Configuration

```python
smm_config = {
    'width': 160,                    # Image width
    'height': 210,                   # Image height
    'input_dim': 5,                  # Input feature dimension
    'slot_dim': 2,                   # Slot latent dimension
    'num_slots': 32,                 # Number of slots
    'use_bias': True,                # Use bias in transformations
    'ns_a': 1.0,                     # Noise scale for mean
    'ns_b': 1.0,                     # Noise scale for bias
    'dof_offset': 10.0,              # Degrees of freedom offset
    'mask_prob': [0.0, 0.0, 0.0, 0.0, 1.0],  # Template probabilities
    'scale': [0.075, 0.075, 0.75, 0.75, 0.75],  # Prior scales
    'transform_inv_v_scale': 100.0,  # Transformation precision
    'bias_inv_v_scale': 0.001,       # Bias precision
    'num_e_steps': 2,                # E-step iterations
    'learning_rate': 1.0,            # Learning rate
    'beta': 0.0,                     # Beta parameter
    'eloglike_threshold': 5.0,       # Growth threshold
    'max_grow_steps': 20             # Maximum growth steps
}
```

### HSMM Configuration (List of Layer Configs)

```python
hsmm_config = [
    {
        'width': 160, 'height': 210, 'input_dim': 5,
        'slot_dim': 2, 'num_slots': 16, 'use_bias': True,
        # ... additional layer 0 config
    },
    {
        'width': 160, 'height': 210, 'input_dim': 5,
        'slot_dim': 4, 'num_slots': 8, 'use_bias': True,
        # ... additional layer 1 config
    },
    {
        'width': 160, 'height': 210, 'input_dim': 5,
        'slot_dim': 6, 'num_slots': 4, 'use_bias': True,
        # ... additional layer 2 config
    }
]
```

### IMM Configuration

```python
imm_config = {
    'num_object_types': 32,          # Number of object identities
    'num_features': 5,               # Feature dimension
    'i_ell_threshold': -500.0,       # Identity threshold
    'cont_scale_identity': 0.5,      # Continuous scale
    'color_precision_scale': 1.0,    # Color precision
    'color_only_identity': False     # Use only color features
}
```

### RMM Configuration

```python
rmm_config = {
    'num_components_per_switch': 25,   # Components per TMM switch
    'num_switches': 100,               # Number of TMM switches
    'num_object_types': 32,            # Object identity types
    'num_features': 5,                 # Feature dimension
    'num_continuous_dims': 7,          # Continuous dimensions
    'interact_with_static': False,     # Static object interaction
    'r_ell_threshold': -100.0,         # Relational threshold
    'i_ell_threshold': -500.0,         # Identity threshold
    'cont_scale_identity': 0.5,        # Identity continuous scale
    'cont_scale_switch': 25.0,         # Switch continuous scale
    'discrete_alphas': [1e-4, 1e-4, 1e-4, 1e-4, 1.0, 1e-4],  # Discrete alphas
    'r_interacting': 0.6,              # Interaction radius
    'r_interacting_predict': 0.6,      # Prediction interaction radius
    'forward_predict': False,          # Forward prediction
    'stable_r': False,                 # Stable relational model
    'relative_distance': True,         # Use relative distances
    'absolute_distance_scale': False,  # Absolute distance scaling
    'reward_prob_threshold': 0.45,     # Reward probability threshold
    'color_precision_scale': 1.0,      # Color precision
    'color_only_identity': False,      # Color-only identity
    'exclude_background': True,        # Exclude background
    'use_ellipses_for_interaction': True,  # Elliptical interactions
    'velocity_scale': 10.0             # Velocity scaling factor
}
```

### TMM Configuration

```python
tmm_config = {
    'n_total_components': 200,        # Total transition components
    'state_dim': 2,                   # State dimension (position)
    'dt': 1.0,                        # Time step
    'vu': 0.05,                       # Unused component probability
    'use_bias': True,                 # Use bias terms
    'use_velocity': True,             # Include velocity modeling
    'sigma_sqr': 2.0,                 # Likelihood variance
    'logp_threshold': -0.00001,       # Log probability threshold
    'position_threshold': 0.15,       # Position change threshold
    'use_unused_counter': True,       # Track unused components
    'clip_value': 5e-4                # Minimum value clipping
}
```

## Output Formats

### Supported Export Formats

- **PNG**: High-quality raster images (default)
- **SVG**: Vector graphics for scalable diagrams
- **PDF**: Publication-quality vector format

### Output Directories

By default, visualizations are saved to the output configuration directory:
```
extra/output/visualizations/
├── static/           # Static plot visualizations
├── animations/       # Animation outputs
└── dashboards/       # Dashboard visualizations
```

## Advanced Usage

### Custom Output Directories

```python
# Specify custom output directory
visualizer = AXIOMModelVisualizer(output_dir="/custom/path/visualizations")
```

### State Evolution Data Format

```python
# SMM evolution data
smm_evolution = [
    {'slots': [[x1, y1], [x2, y2], [x3, y3]]},  # Time step 0
    {'slots': [[x1', y1'], [x2', y2'], [x3', y3']]},  # Time step 1
    # ... more time steps
]

# HSMM evolution data
hsmm_evolution = [
    {
        'layers': [
            {'x': 1.0, 'y': 2.0},
            {'x': 3.0, 'y': 4.0},
            {'x': 5.0, 'y': 6.0}
        ]
    },
    # ... more time steps
]

# RMM evolution data
rmm_evolution = [
    {
        'relations': [
            {'from': [x1, y1], 'to': [x2, y2]},
            {'from': [x3, y3], 'to': [x4, y4]},
        ]
    },
    # ... more time steps
]

# TMM evolution data
tmm_evolution = [
    {'trajectory': [(x1, y1), (x2, y2), (x3, y3)]},
    # ... more time steps
]
```

### Creating Custom Visualizations

```python
# Extend the base visualizer classes for custom diagrams
from model_setup_analysis.model_architecture_visualizer import ArchitectureVisualizer

class CustomVisualizer(ArchitectureVisualizer):
    def visualize_custom_model(self, config, filename):
        """Custom visualization method."""
        if not self.matplotlib_available:
            return None

        fig, ax = self.plt.subplots(figsize=(12, 8))

        # Custom visualization logic here
        # ... draw components, connections, labels

        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        self.plt.close(fig)

        return str(filepath)
```

## Integration with AXIOM Models

### Automatic Configuration Extraction

```python
# Extract configurations from actual AXIOM model instances
from axiom.models.smm import SMMConfig
from axiom.models.imm import IMMConfig
from axiom.models.rmm import RMMConfig
from axiom.models.tmm import TMMConfig

# Convert AXIOM configs to visualization configs
smm_config = SMMConfig(width=160, height=210, input_dim=5).to_dict()
imm_config = IMMConfig(num_object_types=32, num_features=5).to_dict()
# ... etc
```

### Real-time State Visualization

```python
# Visualize model state during training/inference
def visualize_training_progress(model, step, state_data):
    """Create visualization of training progress."""
    visualizer = AXIOMModelVisualizer()

    # Generate state evolution visualization
    evolution_path = visualizer.visualize_state_space_evolution(
        model_type='smm',
        config=model.config,
        evolution_data=state_data,
        filename=f'training_step_{step:04d}.png'
    )

    return evolution_path
```

## File Structure

```
model_setup_analysis/
├── __init__.py
├── model_architecture_visualizer.py     # Main visualization classes
├── model_visualization_demo.py           # Complete demonstration script
└── README_MODEL_VISUALIZATION.md         # This documentation
```

## Dependencies and Requirements

### Required Dependencies

```bash
# Core visualization
matplotlib>=3.5.0
numpy>=1.21.0

# Optional advanced features
networkx>=2.8.0          # Graph visualization
pillow>=9.0.0           # Image processing
```

### System Requirements

- **Python**: 3.7+
- **Memory**: 4GB+ recommended for large visualizations
- **Disk Space**: 100MB+ for storing visualization outputs

## Troubleshooting

### Common Issues

1. **Matplotlib not available**
   ```
   pip install matplotlib
   ```

2. **NetworkX not available** (for relationship diagrams)
   ```
   pip install networkx
   ```

3. **Output directory permission errors**
   ```python
   # Specify a writable output directory
   visualizer = AXIOMModelVisualizer(output_dir="./my_visualizations")
   ```

4. **Memory issues with large models**
   ```python
   # Reduce image resolution
   fig.savefig(filepath, dpi=150, bbox_inches='tight')  # Lower DPI
   ```

### Performance Optimization

- Use vector formats (SVG, PDF) for scalable graphics
- Reduce DPI for faster rendering: `dpi=150` instead of `dpi=300`
- Limit the number of components shown in complex diagrams
- Use `plt.close(fig)` to free memory after saving

## Examples and Gallery

See the `model_visualization_demo.py` script for comprehensive examples of all visualization types.

### Sample Outputs

1. **SMM Architecture**: Shows input → attention → slots → output flow
2. **HSMM Architecture**: Multi-layer hierarchical processing
3. **IMM Architecture**: Identity clustering and recognition
4. **RMM Architecture**: Relational dynamics and predictions
5. **TMM Architecture**: Temporal state transitions
6. **Model Relationships**: Network diagram of inter-model dependencies
7. **Graphical Abstract**: Complete system overview

## Contributing

To add new visualization methods:

1. Extend the base visualizer classes in `model_architecture_visualizer.py`
2. Add new methods to the main `AXIOMModelVisualizer` class
3. Update the demo script with examples
4. Add documentation to this README

### Code Style

- Follow PEP 8 conventions
- Add comprehensive docstrings
- Include type hints
- Handle import errors gracefully
- Use modular, reusable components

## License

This visualization system is part of the AXIOM extensions and follows the same license terms as the main AXIOM project.

## References

- AXIOM: Learning to Play Games in Minutes
- Hierarchical Object-Centric Learning
- Compositional Relational Reasoning
- Temporal Dynamics Modeling

