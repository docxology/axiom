#!/usr/bin/env python3
# Copyright 2025 VERSES AI, Inc.
# Modified by Daniel Ari Friedman in June 2025 using claude-4-sonnet
#
# Initally licensed under the VERSES Academic Research License (the "License");
# you may not use this file except in compliance with the license.
#
# You may obtain a copy of the License at
#
#     https://github.com/VersesTech/axiom/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Configuration optimizations for longer AXIOM runs.

This module provides optimized configurations for extended training runs (100k+ steps)
with considerations for memory usage, computational efficiency, and model pruning.
"""

def get_long_run_args(base_num_steps=100000):
    """
    Get optimized arguments for long AXIOM runs.
    
    Args:
        base_num_steps (int): Base number of steps (default: 100,000 for 10x longer)
        
    Returns:
        list: Command line arguments optimized for long runs
    """
    
    args = [
        f"--num_steps={base_num_steps}",
        
        # Memory and performance optimizations
        "--prune_every=250",  # More frequent pruning (default 500)
        "--bmr_samples=1500",  # Slightly reduced for memory (default 2000)
        "--bmr_pairs=1500",   # Slightly reduced for memory (default 2000)
        
        # Planning optimizations for longer runs
        "--planning_horizon=24",      # Reduced from 32 for efficiency
        "--planning_rollouts=384",    # Reduced from 512 for memory
        "--num_samples_per_rollout=2", # Reduced from 3 for efficiency
        
        # Model capacity adjustments
        "--num_slots=24",           # Reduced from 32 for memory efficiency
        "--n_total_components=400", # Reduced from 500 for memory
        
        # Precision and thresholds
        "--precision_type=float32",  # Ensure float32 for efficiency
        "--smm_eloglike_threshold=6.0",  # Slightly higher threshold
        "--used_threshold=0.025",    # Slightly higher for stability
        
        # Logging optimizations  
        "--velocity_clip_value=0.001",  # Slightly higher for stability
    ]
    
    return args


def get_cpu_optimized_args(base_num_steps=50000):
    """
    Get CPU-optimized arguments for systems without GPU acceleration.
    
    Args:
        base_num_steps (int): Base number of steps (default: 50,000 for CPU)
        
    Returns:
        list: Command line arguments optimized for CPU execution
    """
    
    args = [
        f"--num_steps={base_num_steps}",
        
        # Aggressive optimizations for CPU
        "--prune_every=200",
        "--bmr_samples=1000",
        "--bmr_pairs=1000",
        
        # Minimal planning for CPU efficiency
        "--planning_horizon=16",
        "--planning_rollouts=128",
        "--num_samples_per_rollout=1",
        
        # Reduced model capacity for CPU
        "--num_slots=16",
        "--n_total_components=200",
        
        # CPU-friendly settings
        "--precision_type=float32",
        "--smm_eloglike_threshold=6.5",
        "--used_threshold=0.03",
    ]
    
    return args


def get_memory_efficient_args(base_num_steps=75000):
    """
    Get memory-efficient arguments for systems with limited RAM.
    
    Args:
        base_num_steps (int): Base number of steps (default: 75,000)
        
    Returns:
        list: Command line arguments optimized for memory efficiency
    """
    
    args = [
        f"--num_steps={base_num_steps}",
        
        # Memory-focused optimizations
        "--prune_every=150",  # Very frequent pruning
        "--bmr_samples=800",
        "--bmr_pairs=800",
        
        # Minimal planning footprint
        "--planning_horizon=12",
        "--planning_rollouts=64",
        "--num_samples_per_rollout=1",
        
        # Compact model configuration
        "--num_slots=12",
        "--n_total_components=150",
        "--num_object_types=16",  # Reduced from 32
        
        # Conservative thresholds
        "--smm_eloglike_threshold=7.0",
        "--used_threshold=0.04",
        "--moving_threshold=0.005",
    ]
    
    return args


def get_high_performance_args(base_num_steps=150000):
    """
    Get high-performance arguments for powerful systems with GPU and lots of RAM.
    
    Args:
        base_num_steps (int): Base number of steps (default: 150,000 for 15x longer)
        
    Returns:
        list: Command line arguments optimized for high-performance systems
    """
    
    args = [
        f"--num_steps={base_num_steps}",
        
        # High-performance settings
        "--prune_every=500",  # Standard pruning interval
        "--bmr_samples=3000", # Increased for better model quality
        "--bmr_pairs=3000",
        
        # Enhanced planning
        "--planning_horizon=48",      # Extended horizon
        "--planning_rollouts=768",    # More rollouts
        "--num_samples_per_rollout=4", # More samples
        
        # Expanded model capacity
        "--num_slots=48",           # Increased capacity
        "--n_total_components=750", # Increased capacity
        "--num_object_types=48",    # Increased types
        
        # High-precision settings
        "--precision_type=float32",  # Could use float64 if needed
        "--smm_eloglike_threshold=5.5",  # Lower threshold for sensitivity
        "--used_threshold=0.015",         # Lower threshold
        
        # Fine-tuned parameters
        "--velocity_clip_value=0.0005",
        "--moving_threshold=0.002",
    ]
    
    return args


# Preset configurations for different scenarios
PRESET_CONFIGS = {
    "long_run": {
        "description": "Optimized for 10x longer runs (100k steps)",
        "args_func": get_long_run_args,
        "recommended_for": "Standard extended training"
    },
    "cpu_optimized": {
        "description": "Optimized for CPU-only systems", 
        "args_func": get_cpu_optimized_args,
        "recommended_for": "Systems without GPU acceleration"
    },
    "memory_efficient": {
        "description": "Optimized for limited memory systems",
        "args_func": get_memory_efficient_args, 
        "recommended_for": "Systems with < 16GB RAM"
    },
    "high_performance": {
        "description": "Optimized for powerful systems (15x longer)",
        "args_func": get_high_performance_args,
        "recommended_for": "High-end systems with GPU and 32GB+ RAM"
    }
}


def print_config_info():
    """Print information about available configuration presets."""
    print("Available Configuration Presets:")
    print("=" * 50)
    
    for name, config in PRESET_CONFIGS.items():
        print(f"\n{name}:")
        print(f"  Description: {config['description']}")
        print(f"  Recommended for: {config['recommended_for']}")
        
        # Show sample args
        sample_args = config['args_func']()
        print(f"  Sample args: {' '.join(sample_args[:3])}...")


if __name__ == "__main__":
    print_config_info() 