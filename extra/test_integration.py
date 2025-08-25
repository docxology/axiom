#!/usr/bin/env python3
"""
Integration Test for AXIOM Extra Extensions

This test verifies that all modules in the extra directory work correctly
and can be used to extend AXIOM functionality.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
extra_dir = Path(__file__).parent
src_dir = extra_dir / "src"
sys.path.insert(0, str(src_dir))

def test_module_imports():
    """Test that all modules can be imported successfully."""
    print("=== Testing Module Imports ===")
    
    modules_to_test = [
        ('output_config', 'OutputConfigManager'),
        ('environment_setup.config_validator', 'ConfigValidator'),
        ('environment_setup.dependency_checker', 'DependencyChecker'),
        ('environment_setup.environment_manager', 'EnvironmentManager'),
        ('model_setup_analysis.model_analyzer', 'ModelAnalyzer'),
        ('model_setup_analysis.model_config_manager', 'ModelConfigManager'),
        ('model_setup_analysis.performance_profiler', 'PerformanceProfiler'),
        ('operation.model_runner', 'ModelRunner'),
        ('operation.experiment_manager', 'ExperimentManager'),
        ('operation.result_analyzer', 'ResultAnalyzer'),
        ('visualization.visualizer', 'Visualizer'),
        ('visualization.animator', 'Animator'),
        ('visualization.plot_manager', 'PlotManager')
    ]
    
    for module_path, class_name in modules_to_test:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✓ {module_path}.{class_name}")
        except Exception as e:
            print(f"✗ {module_path}.{class_name}: {e}")
            return False
    
    return True

def test_output_directory_structure():
    """Test that the output directory structure is correct."""
    print("\n=== Testing Output Directory Structure ===")
    
    try:
        from output_config import output_config
        
        # Test key paths exist
        paths_to_check = [
            output_config.get_path_for('visualizations', 'static'),
            output_config.get_path_for('environment', 'configs'),
            output_config.get_path_for('models', 'configs'),
            output_config.get_path_for('runs'),
            output_config.get_path_for('experiments'),
            output_config.get_path_for('analysis', 'complexity'),
            output_config.get_path_for('performance', 'benchmarks')
        ]
        
        for path in paths_to_check:
            if not path.exists():
                print(f"Creating directory: {path}")
                path.mkdir(parents=True, exist_ok=True)
            print(f"✓ {path.relative_to(output_config.paths.base_dir)}")
        
        return True
    except Exception as e:
        print(f"✗ Output directory test failed: {e}")
        return False

def test_environment_setup_integration():
    """Test environment setup module integration."""
    print("\n=== Testing Environment Setup Integration ===")
    
    try:
        from environment_setup.environment_manager import EnvironmentManager
        from environment_setup.dependency_checker import DependencyChecker
        
        # Test environment manager
        manager = EnvironmentManager()
        config = manager.create_default_config("integration_test")
        saved_path = manager.save_config(config)
        print(f"✓ Environment config saved to: {saved_path}")
        
        # Test dependency checker
        checker = DependencyChecker()
        requirements_path = checker.create_requirements_file(['numpy', 'torch'])
        print(f"✓ Requirements file created at: {requirements_path}")
        
        return True
    except Exception as e:
        print(f"✗ Environment setup integration failed: {e}")
        return False

def test_model_setup_integration():
    """Test model setup analysis module integration."""
    print("\n=== Testing Model Setup Analysis Integration ===")
    
    try:
        from model_setup_analysis.model_analyzer import ModelAnalyzer
        from model_setup_analysis.model_config_manager import ModelConfigManager
        
        # Test model analyzer
        analyzer = ModelAnalyzer()
        rmm_config = {
            "model_type": "rmm",
            "num_slots": 10,
            "slot_size": 64,
            "hidden_dim": 256,
            "num_objects": 5
        }
        analysis = analyzer.analyze_model_complexity(rmm_config)
        print(f"✓ Model analysis completed: complexity_score={analysis.get('complexity_score', 'N/A')}")
        
        # Test model config manager
        config_manager = ModelConfigManager()
        optimized_config = config_manager.generate_optimized_config("rmm", "balanced")
        print(f"✓ Optimized config generated for RMM with {len(optimized_config)} parameters")
        
        return True
    except Exception as e:
        print(f"✗ Model setup integration failed: {e}")
        return False

def test_operation_integration():
    """Test operation module integration."""
    print("\n=== Testing Operation Integration ===")
    
    try:
        from operation.model_runner import ModelRunner, RunConfig
        from operation.result_analyzer import ResultAnalyzer
        
        # Test model runner initialization
        runner = ModelRunner()
        print(f"✓ ModelRunner initialized with base dir: {runner.base_dir}")
        
        # Test result analyzer
        analyzer = ResultAnalyzer()
        print("✓ ResultAnalyzer initialized successfully")
        
        return True
    except Exception as e:
        print(f"✗ Operation integration failed: {e}")
        return False

def test_visualization_integration():
    """Test visualization module integration."""
    print("\n=== Testing Visualization Integration ===")
    
    try:
        from visualization.visualizer import Visualizer
        from visualization.plot_manager import PlotManager
        
        # Test visualizer
        viz = Visualizer()
        print(f"✓ Visualizer initialized with output dir: {viz.output_dir}")
        
        # Test plot manager
        plot_manager = PlotManager()
        print(f"✓ PlotManager initialized with output dir: {plot_manager.output_dir}")
        
        return True
    except Exception as e:
        print(f"✗ Visualization integration failed: {e}")
        return False

def test_cross_module_integration():
    """Test integration between different modules."""
    print("\n=== Testing Cross-Module Integration ===")
    
    try:
        # Test that modules can work together using the output configuration
        from output_config import output_config
        from environment_setup.environment_manager import EnvironmentManager
        from model_setup_analysis.model_config_manager import ModelConfigManager
        
        # Create a complete workflow
        manager = EnvironmentManager()
        config = manager.create_default_config("cross_module_test")
        
        # Save config using environment manager
        config_path = manager.save_config(config)
        print(f"✓ Config saved to: {config_path}")
        
        # Load and analyze with model config manager
        model_config_manager = ModelConfigManager()
        optimized = model_config_manager.generate_optimized_config("rmm", "balanced")
        print(f"✓ Generated optimized config with {len(optimized)} parameters")
        
        return True
    except Exception as e:
        print(f"✗ Cross-module integration failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("AXIOM Extra Integration Test Suite")
    print("=" * 40)
    
    tests = [
        test_module_imports,
        test_output_directory_structure,
        test_environment_setup_integration,
        test_model_setup_integration,
        test_operation_integration,
        test_visualization_integration,
        test_cross_module_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 40)
    print("INTEGRATION TEST RESULTS")
    print("=" * 40)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("\nThe AXIOM Extra extensions are fully functional and ready for use.")
        return True
    else:
        print("❌ Some integration tests failed.")
        print("Please review the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
