# AXIOM Extended Training Guide

This guide covers how to run AXIOM for extended training sessions (10x longer or more) with optimal configurations for different hardware setups.

## Quick Start for 10x Longer Training

Run the Explode game for 100,000 steps (10x longer than default) with optimizations:

```bash
python run_all.py --config_preset long_run --games Explode
```

Run all games with 10x longer training:

```bash
python run_all.py --config_preset long_run
```

## Configuration Presets

### Long Run (Recommended for 10x Longer)
- **Steps**: 100,000 (10x default)
- **Optimizations**: Balanced memory/performance
- **Use case**: Standard extended training

```bash
python run_all.py --config_preset long_run --games Explode,Bounce
```

### CPU Optimized
- **Steps**: 50,000 (5x default)
- **Optimizations**: Minimal computational requirements
- **Use case**: Systems without GPU

```bash
python run_all.py --config_preset cpu_optimized
```

### Memory Efficient
- **Steps**: 75,000 (7.5x default)
- **Optimizations**: Reduced memory footprint
- **Use case**: Systems with < 16GB RAM

```bash
python run_all.py --config_preset memory_efficient --parallel --max_workers 2
```

### High Performance
- **Steps**: 150,000 (15x default)
- **Optimizations**: Maximum model capacity
- **Use case**: High-end systems with GPU and 32GB+ RAM

```bash
python run_all.py --config_preset high_performance --timeout_hours 8
```

## Custom Configurations

You can combine presets with custom arguments:

```bash
# Long run with custom step count
python run_all.py --config_preset long_run --num_steps 200000 --games Explode

# Memory efficient with custom planning
python run_all.py --config_preset memory_efficient --planning_horizon 8 --games Jump,Hunt

# CPU optimized with specific games
python run_all.py --config_preset cpu_optimized --games Fruits,Gold,Impact
```

## Optimization Details

### Long Run Optimizations
- More frequent model pruning (`--prune_every=250` vs default 500)
- Reduced BMR samples for memory efficiency
- Smaller planning horizon for faster decisions
- Optimized model capacity for stability

### Key Parameters Adjusted
- `prune_every`: 250 (more frequent cleanup)
- `bmr_samples`: 1500 (reduced from 2000)
- `planning_horizon`: 24 (reduced from 32)
- `planning_rollouts`: 384 (reduced from 512)
- `num_slots`: 24 (reduced from 32)

## Performance Monitoring

### Expected Timing
- **Short run (10k steps)**: ~2-3 minutes per game
- **Long run (100k steps)**: ~20-30 minutes per game
- **High performance (150k steps)**: ~45-60 minutes per game

### Memory Usage
- **Standard**: ~4-6GB RAM
- **Long run**: ~6-8GB RAM
- **High performance**: ~10-16GB RAM

### Disk Space
Results include CSV files, MP4 videos, and wandb logs:
- **Per game**: ~50-100MB
- **All 10 games (long run)**: ~500MB-1GB

## Parallel Execution

For faster completion, run games in parallel:

```bash
# Run 4 games simultaneously (requires adequate RAM)
python run_all.py --config_preset long_run --parallel --max_workers 4

# Conservative parallel execution for limited systems
python run_all.py --config_preset memory_efficient --parallel --max_workers 2
```

**Warning**: Parallel execution significantly increases memory usage. Monitor system resources.

## Troubleshooting

### Out of Memory
1. Use `memory_efficient` preset
2. Reduce `--max_workers` if using parallel
3. Close other applications

### Slow Performance
1. Use `cpu_optimized` preset for CPU-only systems
2. Reduce `--num_steps` for testing
3. Run fewer games simultaneously

### Timeout Issues
1. Increase `--timeout_hours` for very long runs
2. Monitor system performance during execution

## Advanced Usage

### Custom Step Counts
```bash
# 20x longer than default
python run_all.py --config_preset high_performance --num_steps 200000

# Custom step count with optimizations
python run_all.py --config_preset long_run --num_steps 500000 --timeout_hours 12
```

### System-Specific Tuning
```bash
# For 8GB RAM systems
python run_all.py --config_preset memory_efficient --num_slots 16 --n_total_components 200

# For high-end GPU systems
python run_all.py --config_preset high_performance --bmr_samples 4000 --precision_type float32
```

### Batch Processing
```bash
# Run subset of games first
python run_all.py --config_preset long_run --games Explode,Bounce,Cross

# Then run remaining games
python run_all.py --config_preset long_run --games Drive,Fruits,Gold,Hunt,Impact,Jump
```

## Results Analysis

All results are automatically logged to:
- **CSV files**: `{game}.csv` with step-by-step metrics
- **Video files**: `{game}.mp4` with gameplay recordings
- **Wandb logs**: Online dashboard with real-time metrics
- **Summary files**: `run_all_results_*.txt` with execution summary

## Best Practices

1. **Start Small**: Test with 1-2 games before running all 10
2. **Monitor Resources**: Check RAM/CPU usage during execution
3. **Use Presets**: Leverage optimized configurations for your system
4. **Plan Timing**: Long runs can take several hours
5. **Check Results**: Verify wandb logging is working before long runs

## Example Workflows

### Research/Development
```bash
# Quick validation
python run_all.py --config_preset long_run --games Explode --num_steps 1000 --dry_run

# Single game extended training
python run_all.py --config_preset long_run --games Explode

# Full evaluation
python run_all.py --config_preset long_run
```

### Production/Benchmarking
```bash
# High-quality runs for publication
python run_all.py --config_preset high_performance --timeout_hours 10

# Comprehensive evaluation with parallel execution
python run_all.py --config_preset long_run --parallel --max_workers 3
``` 