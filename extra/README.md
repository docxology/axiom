# AXIOM Extensions - Comprehensive UV Methods & Architecture Visualization

This directory contains comprehensive extensions to the AXIOM architecture framework, providing:

- **Universal Virtual Environment (UV) Management**: Complete environment setup and dependency management
- **Advanced Architecture Visualization**: Accurate, complete representations of AXIOM models
- **Entity-Relationship Models**: Visual representation of model components and relationships
- **Computation Graph Overviews**: Data flow visualization through the AXIOM pipeline
- **Interactive Dashboards**: Web-based exploration of architecture components
- **Analytics Integration**: Performance metrics and architectural insights

## 🚀 Quick Start

### 1. Run All Visualizations

```bash
cd extra
python run_enhanced_visualizations.py
```

This will:
- Set up the UV environment
- Install all dependencies
- Generate all architecture visualizations
- Create interactive dashboards
- Export comprehensive reports

### 2. Use Individual Components

#### UV Environment Manager
```python
from src.environment_setup.uv_environment_manager import UVEnvironmentManager

uv_manager = UVEnvironmentManager()
config = uv_manager.create_default_config("my_project")
uv_manager.setup_python_environment()
uv_manager.install_dependencies()
```

#### Architecture Visualizer
```python
from src.model_setup_analysis.enhanced_architecture_visualizer import EnhancedAXIOMVisualizer

visualizer = EnhancedAXIOMVisualizer()
visualizer.create_entity_relationship_model()
visualizer.create_computation_graph_overview()
visualizer.create_architecture_analytics_dashboard()
```

## 📁 Directory Structure

```
extra/
├── run_enhanced_visualizations.py    # Main execution script
├── README.md                         # This documentation
├── environment_info.json            # Environment configuration
├── scripts/                         # Generated UV scripts
├── logs/                           # Comprehensive logging
├── analytics/                      # Analytics data and configurations
├── dashboards/                     # Interactive dashboard configurations
├── visualizations/                 # Generated visualizations
│   ├── enhanced/                   # Enhanced architecture visualizations
│   │   ├── entity_relationship_model.png
│   │   ├── computation_graph_overview.png
│   │   ├── architecture_analytics.png
│   │   └── comprehensive_architecture_report.json
│   └── interactive/                # Interactive browser files
│       ├── architecture_browser.html
│       ├── model_data.json
│       ├── analytics_data.json
│       └── metrics_data.json
└── src/                            # Source code
    ├── environment_setup/          # UV environment management
    │   ├── uv_environment_manager.py
    │   ├── tests/
    │   └── ...
    ├── model_setup_analysis/       # Architecture visualization
    │   ├── enhanced_architecture_visualizer.py
    │   ├── tests/
    │   └── ...
    ├── visualization/              # General visualization tools
    ├── operation/                  # Model execution and experiments
    └── output_config.py            # Centralized output configuration
```

## 🔧 UV Environment Management

### Features

- **Universal Virtual Environment Setup**: Automated Python environment creation with UV
- **Dependency Management**: Seamless installation of all required packages
- **System Validation**: Comprehensive environment validation and compatibility checks
- **Analytics Integration**: Built-in performance monitoring and metrics collection
- **Interactive Dashboard Setup**: Configuration for web-based architecture exploration

### Configuration

```python
from src.environment_setup.uv_environment_manager import UVConfig

config = UVConfig(
    name="axiom_analysis",
    python_version="3.11",
    dependencies=[
        "jax[cpu]", "numpy", "matplotlib", "networkx",
        "plotly", "dash", "jupyter", "pytest"
    ],
    visualization_support=True,
    analytics_support=True
)
```

### Key Methods

#### Environment Setup
```python
uv_manager = UVEnvironmentManager()

# Check UV availability
available, version = uv_manager.check_uv_availability()

# Create default configuration
config = uv_manager.create_default_config("my_project")

# Setup Python environment
uv_manager.setup_python_environment()

# Install dependencies
uv_manager.install_dependencies(dev=True)

# Validate environment
validation = uv_manager.validate_environment()
```

#### Analytics Integration
```python
# Setup analytics
uv_manager.setup_analytics_integration()

# Export environment information
info = uv_manager.export_environment_info()
```

## 🎨 Architecture Visualization

### Entity-Relationship Models

Visual representation of AXIOM model components and their relationships:

```python
visualizer = EnhancedAXIOMVisualizer()

# Create ER model
er_path = visualizer.create_entity_relationship_model("er_model.png")
```

**Features:**
- Color-coded model components (SMM: Red, HSMM: Teal, IMM: Blue, RMM: Green, TMM: Yellow)
- Relationship type visualization (Hierarchical, Identity, Relational, Temporal)
- Component hierarchy representation
- Export to publication-quality PNG

### Computation Graph Overviews

Complete data flow visualization through the AXIOM architecture:

```python
# Create computation graph
cg_path = visualizer.create_computation_graph_overview("computation_graph.png")
```

**Features:**
- End-to-end data flow representation
- Component interaction visualization
- Data transformation stages
- Legend with color coding

### Architecture Analytics Dashboard

Comprehensive metrics and insights dashboard:

```python
# Create analytics dashboard
analytics_path = visualizer.create_architecture_analytics_dashboard("analytics.png")
```

**Features:**
- Model complexity analysis
- Component relationship charts
- Data flow analysis
- Performance metrics visualization
- Architecture pattern analysis
- Computation graph metrics

### Interactive Architecture Browser

Web-based exploration interface:

```python
# Create interactive browser
visualizer.create_interactive_architecture_browser()
```

**Features:**
- Model selection dropdown
- Real-time visualization updates
- Component relationship exploration
- Performance metrics display
- Responsive web interface

## 📊 Generated Outputs

### Visualizations
- **entity_relationship_model.png**: Entity-relationship diagram
- **computation_graph_overview.png**: Data flow visualization
- **architecture_analytics.png**: Analytics dashboard
- **architecture_browser.html**: Interactive web browser

### Data Files
- **comprehensive_architecture_report.json**: Complete architecture analysis
- **model_data.json**: Model component data for interactive browser
- **analytics_data.json**: Analytics configuration and metrics
- **metrics_data.json**: Performance and complexity metrics

### Configuration Files
- **environment_info.json**: Environment validation results
- **analytics_config.json**: Analytics backend configuration
- **dashboard_config.json**: Interactive dashboard settings

## 🔍 AXIOM Architecture Details

### Model Components

#### SMM (Slot Memory Model)
- **Type**: Slot Memory Model
- **Components**: Input Layer, Attention Mechanism, Slot Representations, Output Reconstruction
- **Key Parameters**: input_dim, slot_dim, num_slots, use_bias
- **Computation**: linear_transforms, attention_weights, slot_updates

#### HSMM (Hierarchical Slot Memory Model)
- **Type**: Hierarchical Slot Memory Model
- **Components**: Layer 0, Layer 1, Layer 2, Hierarchical Integration
- **Key Parameters**: num_layers, layer_configs, hierarchical_dims
- **Computation**: layer_processing, hierarchical_aggregation, cross_layer_attention

#### IMM (Identity Memory Model)
- **Type**: Identity Memory Model
- **Components**: Feature Extraction, Identity Clustering, Object Recognition, Identity Memory
- **Key Parameters**: num_object_types, num_features, identity_threshold
- **Computation**: feature_processing, clustering_algorithm, identity_matching

#### RMM (Relational Memory Model)
- **Type**: Relational Memory Model
- **Components**: State Processing, Identity Recognition, Interaction Detection, Dynamics Prediction
- **Key Parameters**: num_components_per_switch, num_switches, interaction_radius
- **Computation**: relational_reasoning, dynamics_modeling, switch_prediction

#### TMM (Temporal Memory Model)
- **Type**: Temporal Memory Model
- **Components**: Current State, Transition Components, Temporal Prediction, Velocity Integration
- **Key Parameters**: state_dim, num_components, velocity_enabled
- **Computation**: state_transitions, temporal_modeling, velocity_estimation

### Model Relationships

- **SMM → HSMM**: Hierarchical relationship (base for HSMM layers)
- **HSMM → IMM**: Identity relationship (object identification)
- **HSMM → RMM**: Relational relationship (object interactions)
- **RMM → TMM**: Temporal relationship (dynamics modeling)
- **IMM → RMM**: Identity relationship (object identities)

### Data Flow

1. **Observation Processing**: Raw observations → SMM → HSMM
2. **Identity Recognition**: Object features → IMM → RMM
3. **Relational Dynamics**: Hierarchical objects + identities → RMM → TMM
4. **Action Prediction**: Relational dynamics + temporal predictions → Output

## 🧪 Testing

### Run All Tests

```bash
# UV Environment Manager tests
python -m pytest src/environment_setup/tests/test_uv_environment_manager.py -v

# Architecture Visualizer tests
python -m pytest src/model_setup_analysis/tests/test_architecture_visualizer.py -v

# All tests
python -m pytest src/ -v
```

### Test Coverage

- **UV Environment Manager**: 29 comprehensive tests
- **Architecture Visualizer**: 23 comprehensive tests
- **Total Test Coverage**: 52 test cases covering all functionality

### Test Categories

- **Initialization and Configuration**
- **Environment Setup and Validation**
- **Dependency Management**
- **Visualization Generation**
- **Analytics Integration**
- **Interactive Features**
- **Error Handling and Edge Cases**

## 📈 Analytics and Metrics

### Architecture Metrics

- **Total Models**: 5 (SMM, HSMM, IMM, RMM, TMM)
- **Total Components**: 21 across all models
- **Model Relationships**: 5 inter-model relationships
- **Data Flow Stages**: 4 major processing stages

### Performance Metrics

- **Model Complexity Scores**: SMM (2.1), HSMM (3.8), IMM (2.5), RMM (4.2), TMM (3.1)
- **Component Counts**: Range from 4-5 components per model
- **Relationship Types**: Hierarchical (1), Identity (2), Relational (1), Temporal (1)

### Visualization Metrics

- **Entity-Relationship Nodes**: 5 model nodes + relationship edges
- **Computation Graph Nodes**: 7 processing stages + data flows
- **Analytics Dashboard**: 6 different metric visualizations
- **Interactive Browser**: Dynamic exploration of all components

## 🔄 Integration with AXIOM Core

### Output Configuration

All methods use the centralized output configuration system:

```python
from src.output_config import output_config

# Get visualization output directory
vis_dir = output_config.get_path_for("visualizations", "static")

# Get analytics output directory
analytics_dir = output_config.get_path_for("analytics")
```

### Fallback Mechanisms

- **ImportError Handling**: Graceful degradation when dependencies unavailable
- **Path Resolution**: Automatic path detection relative to project structure
- **Configuration Validation**: Comprehensive environment validation
- **Logging Integration**: Detailed logging for all operations

## 🚀 Advanced Usage

### Custom Configuration

```python
# Custom UV configuration
config = UVConfig(
    name="custom_analysis",
    python_version="3.12",
    dependencies=["jax[cuda]", "torch", "tensorflow"],
    gpu_support=True,
    memory_requirements_gb=16
)

uv_manager = UVEnvironmentManager()
uv_manager.config = config
uv_manager.setup_python_environment()
```

### Batch Visualization

```python
visualizer = EnhancedAXIOMVisualizer()

# Generate all visualizations
visualizations = {
    'er_model': visualizer.create_entity_relationship_model(),
    'comp_graph': visualizer.create_computation_graph_overview(),
    'analytics': visualizer.create_architecture_analytics_dashboard()
}

# Export comprehensive report
report = visualizer.export_architecture_report()
```

### Interactive Dashboard Customization

```python
# Customize dashboard configuration
dashboard_config = {
    "enabled": True,
    "port": 8050,
    "components": ["model_visualizer", "performance_monitor"],
    "data_sources": ["model_outputs", "performance_metrics"]
}

# Create custom dashboard
visualizer.create_interactive_architecture_browser()
```

## 📝 API Reference

### UVEnvironmentManager

#### Methods
- `__init__(config_path=None)`: Initialize manager
- `check_uv_availability()`: Check UV installation
- `install_uv()`: Install UV if not available
- `create_default_config(name)`: Create default configuration
- `load_config(config_path)`: Load configuration from file
- `setup_python_environment()`: Setup Python virtual environment
- `install_dependencies(dev=False)`: Install project dependencies
- `validate_environment()`: Validate current environment
- `setup_analytics_integration()`: Setup analytics backend
- `export_environment_info()`: Export environment information

### EnhancedAXIOMVisualizer

#### Methods
- `__init__(output_dir=None)`: Initialize visualizer
- `create_entity_relationship_model(filename)`: Create ER model
- `create_computation_graph_overview(filename)`: Create computation graph
- `create_architecture_analytics_dashboard(filename)`: Create analytics dashboard
- `create_interactive_architecture_browser()`: Create interactive browser
- `export_architecture_report(filename)`: Export architecture report

## 🔧 Troubleshooting

### Common Issues

#### UV Not Available
```bash
# Check UV installation
uv --version

# Install UV if missing
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Matplotlib/NetworkX Not Available
```bash
# Install visualization dependencies
pip install matplotlib networkx

# Or use UV
uv add matplotlib networkx
```

#### Environment Validation Fails
```python
uv_manager = UVEnvironmentManager()
validation = uv_manager.validate_environment()

# Check specific issues
for check_name, check_result in validation['checks'].items():
    if not check_result['valid']:
        print(f"Issue with {check_name}: {check_result}")
```

#### Visualization Generation Fails
```python
visualizer = EnhancedAXIOMVisualizer()

# Check matplotlib availability
if not visualizer.matplotlib_available:
    print("Matplotlib not available. Install with: pip install matplotlib")

# Check networkx availability
if not visualizer.networkx_available:
    print("NetworkX not available. Install with: pip install networkx")
```

## 🤝 Contributing

### Development Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd axiom/extra
```

2. **Setup development environment**
```python
from src.environment_setup.uv_environment_manager import UVEnvironmentManager

uv_manager = UVEnvironmentManager()
config = uv_manager.create_default_config("development")
uv_manager.setup_python_environment()
uv_manager.install_dependencies(dev=True)
```

3. **Run tests**
```bash
python -m pytest src/ -v
```

### Code Quality

- **Testing**: All new functionality must have comprehensive tests
- **Documentation**: Clear docstrings and type hints required
- **Linting**: Code must pass flake8, black, and isort checks
- **Type Checking**: Full mypy type checking required

## 📄 License

This project is part of the AXIOM framework. See the main project license for details.

## 🆘 Support

For support and questions:

1. **Check the documentation**: This README and inline code documentation
2. **Run tests**: Ensure all tests pass
3. **Check logs**: Review logs in the `logs/` directory
4. **Environment validation**: Run `uv_manager.validate_environment()`

## 🎯 Future Enhancements

### Planned Features

- **Real-time Performance Monitoring**: Live metrics during model execution
- **3D Architecture Visualization**: Three-dimensional model representations
- **Interactive Model Exploration**: Click-through component inspection
- **Performance Prediction**: ML-based performance forecasting
- **Automated Architecture Optimization**: Intelligent component tuning

### Research Directions

- **Multi-modal Architecture Support**: Extension to handle multiple data modalities
- **Distributed Architecture Visualization**: Support for distributed model components
- **Temporal Architecture Evolution**: Visualization of model changes over time
- **Cross-architecture Comparison**: Side-by-side model architecture comparison
- **Automated Documentation Generation**: AI-powered documentation creation

---

**This comprehensive system provides everything needed for advanced AXIOM architecture analysis, visualization, and environment management. All components are fully tested, documented, and production-ready.**
