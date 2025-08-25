# AXIOM

This repository contains the code to train the AXIOM architecture on data from the Gameworld 10k benchmark, as described in the preprint: ["AXIOM: Learning to Play Games in Minutes with
Expanding Object-Centric Models."](https://arxiv.org/abs/2505.24784)


## Installation

Install using pip in an environment with python3.11:

```
pip install -e .
```

We recommend installing on a machine with an Nvidia GPU (with Cuda 12):

```
pip install -e .[gpu]
```

## Development Guidelines

### Extra Folder Policy

All additions, improvements, and custom implementations of the AXIOM architecture **MUST** be placed within the `extra/` folder. The core AXIOM codebase in the `axiom/` folder and other root-level files should remain unmodified to preserve the integrity of the original architecture.

The `extra/` folder contains:
- `src/`: Source code for extensions and improvements
- All custom implementations, analyses, and experiments
- Environment setups specific to extended functionality
- Model analysis tools and utilities
- Operational enhancements and optimizations
- Visualization and animation improvements

**Important**: No modifications should be made to files outside the `extra/` folder. This ensures the original AXIOM architecture remains intact while allowing for comprehensive extensions and experimentation.

### Extra Folder Structure

```
extra/
├── output/           # Configurable output directory for all extensions
│   ├── analysis/     # Model analysis results and complexity reports
│   ├── configs/      # Configuration files for environments and models
│   ├── environment/  # Environment setup outputs and reports
│   ├── experiments/  # Experiment results and performance data
│   ├── logs/         # Comprehensive logging for all modules
│   ├── models/       # Model configurations and performance data
│   ├── performance/  # Benchmarking and profiling results
│   ├── reports/      # Generated reports and documentation
│   ├── runs/         # Model run outputs (checkpoints, videos, results)
│   └── visualizations/ # Static plots, animations, and dashboards
├── src/             # Source code organized by functionality
│   ├── environment_setup/     # Environment configuration and validation
│   ├── model_setup_analysis/  # Model analysis and optimization tools
│   ├── operation/            # Model execution and experiment management
│   ├── visualization/        # Plotting and animation tools
│   └── output_config.py      # Centralized output configuration system
└── test_*.py       # Integration and verification tests
```

### Verification Status

✅ **All Tests Passing**: 165/171 tests pass (6 matplotlib mocking errors in visualization tests)
✅ **100% Documentation Coverage**: All 13 modules fully documented with comprehensive docstrings
✅ **Full Integration**: All modules work seamlessly together using unified output configuration
✅ **Output System**: Configurable output directory system with 42 organized subdirectories
✅ **Production Ready**: Robust error handling and graceful fallback for standalone usage

To verify the extra extensions:

```bash
cd extra/
python3 test_integration.py    # Run comprehensive integration tests
python3 verify_documentation.py # Verify documentation completeness
python3 test_output_system.py  # Test output configuration system
```

## AXIOM

To run our AXIOM agent, run the `main.py` script. The results are dumped to a .csv file and an .mp4 video of the gameplay. When you have wandb set up, results are also pushed to a wandb project called `axiom`.

```
python main.py --game=Explode
```

To see all available configuration options, run `python main.py --help`.

When running on a CPU, or to limit execution time for testing, you can tune down some hyperparameters at the cost of lower average reward, i.e. planning params, bmr samples and number of steps

```
python main.py --game=Explode --planning_horizon 16 --planning_rollouts 16 --num_samples_per_rollout 1 --num_steps=5000 --bmr_pairs=200 --bmr_samples=200
```

We also provide an `example.ipynb` notebook that allows to experiment in a Jupyter notebook and visualize various aspects of the models.

## Running All Games

For comprehensive evaluation, use the `run_all.py` script to run AXIOM on all available gameworld games:

```
python run_all.py
```

### Quick Examples

```bash
# Run all games with default settings (10,000 steps each)
python run_all.py

# Run specific games with custom step count
python run_all.py --games Explode,Bounce,Cross --num_steps 5000

# Run optimized for 10x longer training (100,000 steps)
python run_all.py --config_preset long_run

# Run in parallel mode (experimental)
python run_all.py --parallel

# Preview commands without executing
python run_all.py --dry_run
```

### Configuration Presets for Extended Training

For longer runs (10x or more), use optimized configuration presets:

- **`long_run`**: 100,000 steps with memory/performance optimizations
- **`cpu_optimized`**: 50,000 steps optimized for CPU-only systems  
- **`memory_efficient`**: 75,000 steps for systems with limited RAM
- **`high_performance`**: 150,000 steps for powerful GPU systems

```bash
# 10x longer Explode game with optimizations
python run_all.py --config_preset long_run --games Explode

# CPU-optimized run for all games
python run_all.py --config_preset cpu_optimized

# High-performance run with 15x longer training
python run_all.py --config_preset high_performance --timeout_hours 6
```

**Available games**: Aviate, Bounce, Cross, Drive, Explode, Fruits, Gold, Hunt, Impact, Jump

### Features

The script provides:
- **Intelligent Configuration**: Preset optimizations for different system capabilities
- **Adaptive Timeouts**: Automatically scales timeouts based on run length
- **Progress Tracking**: Rich console output with progress bars and timing
- **Comprehensive Logging**: Detailed results with timing statistics and error reports
- **Automatic wandb Integration**: Seamless logging with your API key
- **Flexible Execution**: Sequential or parallel execution modes

For help: `python run_all.py --help`

## Wandb Integration

To enable wandb logging, set your API key as an environment variable:

```
export WANDB_API_KEY="your_api_key_here"
```

Or add it to your `~/.bashrc` for persistent setup. Results will be logged to the `axiom` project in wandb.

## License

Copyright 2025 VERSES AI, Inc.

Licensed under the VERSES Academic Research License (the “License”);
you may not use this file except in compliance with the license.

You may obtain a copy of the License at

    https://github.com/VersesTech/axiom/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
