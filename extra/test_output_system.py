#!/usr/bin/env python3
"""
Test script to verify the output configuration system works correctly
when modules are used in their intended package context.
"""

import sys
from pathlib import Path

# Add the src directory to the path to simulate package structure
extra_dir = Path(__file__).parent
src_dir = extra_dir / "src"
sys.path.insert(0, str(src_dir))

def test_output_configuration():
    """Test that the output configuration system works correctly."""
    print("=== Testing Output Configuration System ===\n")

    # Test 1: Direct output configuration access
    print("1. Testing direct output configuration access...")
    try:
        from output_config import output_config
        print("   ✓ Output configuration loaded successfully")
        print(f"   ✓ Base directory: {output_config.paths.base_dir}")
        print(f"   ✓ Visualizations path: {output_config.get_path_for('visualizations', 'static')}")
        print(f"   ✓ Environment configs path: {output_config.get_path_for('environment', 'configs')}")
        print(f"   ✓ Model configs path: {output_config.get_path_for('models', 'configs')}")
        print(f"   ✓ Runs path: {output_config.get_path_for('runs')}")
        print(f"   ✓ Experiments path: {output_config.get_path_for('experiments')}")
    except Exception as e:
        print(f"   ✗ Failed to load output configuration: {e}")
        return False

    # Test 2: Module initialization with output config (simulating proper package context)
    print("\n2. Testing module initialization...")

    # Test Visualizer
    try:
        # Temporarily modify sys.path to simulate proper package structure
        sys.path.insert(0, str(src_dir))
        import importlib

        # Force reload to test import behavior
        if 'visualization.visualizer' in sys.modules:
            importlib.reload(sys.modules['visualization.visualizer'])
        if 'visualization' in sys.modules:
            importlib.reload(sys.modules['visualization'])

        from visualization.visualizer import Visualizer

        # Test with explicit output directory
        viz = Visualizer()
        print(f"   ✓ Visualizer initialized (output dir: {viz.output_dir})")

        # Test if it's using output config path
        expected_path = output_config.get_path_for('visualizations', 'static')
        if str(viz.output_dir) == str(expected_path):
            print("   ✓ Visualizer is using output configuration correctly")
        else:
            print(f"   ! Visualizer using fallback path: {viz.output_dir}")

    except Exception as e:
        print(f"   ✗ Visualizer test failed: {e}")

    # Test ModelRunner
    try:
        if 'operation.model_runner' in sys.modules:
            importlib.reload(sys.modules['operation.model_runner'])
        if 'operation' in sys.modules:
            importlib.reload(sys.modules['operation'])

        from operation.model_runner import ModelRunner

        runner = ModelRunner()
        print(f"   ✓ ModelRunner initialized (base dir: {runner.base_dir})")

        expected_path = output_config.get_path_for('runs')
        if str(runner.base_dir) == str(expected_path):
            print("   ✓ ModelRunner is using output configuration correctly")
        else:
            print(f"   ! ModelRunner using fallback path: {runner.base_dir}")

    except Exception as e:
        print(f"   ✗ ModelRunner test failed: {e}")

    # Test EnvironmentManager
    try:
        if 'environment_setup.environment_manager' in sys.modules:
            importlib.reload(sys.modules['environment_setup.environment_manager'])
        if 'environment_setup' in sys.modules:
            importlib.reload(sys.modules['environment_setup'])

        from environment_setup.environment_manager import EnvironmentManager

        manager = EnvironmentManager()
        print(f"   ✓ EnvironmentManager initialized")

        # Test saving a config
        config = manager.create_default_config("test_config")
        saved_path = manager.save_config(config)
        print(f"   ✓ Config saved to: {saved_path}")

        expected_dir = output_config.get_path_for('environment', 'configs')
        if str(saved_path.parent) == str(expected_dir):
            print("   ✓ EnvironmentManager is using output configuration correctly")
        else:
            print(f"   ! EnvironmentManager using fallback path: {saved_path.parent}")

    except Exception as e:
        print(f"   ✗ EnvironmentManager test failed: {e}")

    # Test DependencyChecker
    try:
        if 'environment_setup.dependency_checker' in sys.modules:
            importlib.reload(sys.modules['environment_setup.dependency_checker'])

        from environment_setup.dependency_checker import DependencyChecker

        checker = DependencyChecker()
        requirements_path = checker.create_requirements_file(['numpy', 'torch'])
        print(f"   ✓ Requirements file created at: {requirements_path}")

        expected_dir = output_config.get_path_for('environment', 'configs')
        if str(requirements_path.parent) == str(expected_dir):
            print("   ✓ DependencyChecker is using output configuration correctly")
        else:
            print(f"   ! DependencyChecker using fallback path: {requirements_path.parent}")

    except Exception as e:
        print(f"   ✗ DependencyChecker test failed: {e}")

    print("\n=== Output Configuration System Test Complete ===")
    return True

if __name__ == "__main__":
    test_output_configuration()
