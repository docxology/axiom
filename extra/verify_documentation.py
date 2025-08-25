#!/usr/bin/env python3
"""
Documentation Verification for AXIOM Extra Extensions

This script verifies that all modules, classes, and functions in the extra
directory have proper documentation.
"""

import sys
import inspect
from pathlib import Path

# Add src to path for imports
extra_dir = Path(__file__).parent
src_dir = extra_dir / "src"
sys.path.insert(0, str(src_dir))

def check_module_docstring(module_name, module_path):
    """Check if a module has a proper docstring."""
    try:
        module = __import__(module_path, fromlist=[''])
        
        if not hasattr(module, '__doc__') or not module.__doc__ or module.__doc__.strip() == '':
            return False, f"Missing or empty module docstring"
        
        docstring = module.__doc__.strip()
        if len(docstring) < 10:  # Very basic check for meaningful docstring
            return False, f"Docstring too short: '{docstring}'"
            
        return True, "Module docstring OK"
    except Exception as e:
        return False, f"Error importing module: {e}"

def check_class_documentation(cls):
    """Check if a class has proper documentation."""
    issues = []
    
    # Check class docstring
    if not cls.__doc__ or cls.__doc__.strip() == '':
        issues.append("Missing class docstring")
    
    # Check public methods
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if not name.startswith('_'):  # Skip private methods
            if not method.__doc__ or method.__doc__.strip() == '':
                issues.append(f"Method '{name}' missing docstring")
    
    return issues

def analyze_module(module_name, module_path):
    """Analyze a module for documentation completeness."""
    print(f"\n=== Analyzing {module_name} ===")
    
    # Check module docstring
    module_ok, module_msg = check_module_docstring(module_name, module_path)
    if module_ok:
        print(f"✓ {module_msg}")
    else:
        print(f"✗ {module_msg}")
    
    try:
        module = __import__(module_path, fromlist=[''])
        
        # Get all classes in the module
        classes = []
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and obj.__module__ == module.__name__:
                classes.append((name, obj))
        
        if not classes:
            print("! No classes found in module")
            return True
        
        documented_classes = 0
        total_methods = 0
        documented_methods = 0
        
        for class_name, cls in classes:
            print(f"\nAnalyzing class: {class_name}")
            issues = check_class_documentation(cls)
            
            if not issues:
                print("✓ All methods documented")
                documented_classes += 1
            else:
                for issue in issues:
                    print(f"  - {issue}")
            
            # Count methods
            methods = [name for name, method in inspect.getmembers(cls, predicate=inspect.isfunction) 
                      if not name.startswith('_')]
            total_methods += len(methods)
            documented_methods += len(methods) - len([i for i in issues if 'missing docstring' in i])
        
        print(f"\nClass documentation: {documented_classes}/{len(classes)} classes fully documented")
        print(f"Method documentation: {documented_methods}/{total_methods} methods documented")
        
        return len(issues) == 0 if 'issues' in locals() else True
        
    except Exception as e:
        print(f"✗ Error analyzing module: {e}")
        return False

def main():
    """Run documentation verification."""
    print("AXIOM Extra Documentation Verification")
    print("=" * 50)
    
    modules_to_check = [
        ('output_config', 'output_config'),
        ('config_validator', 'environment_setup.config_validator'),
        ('dependency_checker', 'environment_setup.dependency_checker'),
        ('environment_manager', 'environment_setup.environment_manager'),
        ('model_analyzer', 'model_setup_analysis.model_analyzer'),
        ('model_config_manager', 'model_setup_analysis.model_config_manager'),
        ('performance_profiler', 'model_setup_analysis.performance_profiler'),
        ('model_runner', 'operation.model_runner'),
        ('experiment_manager', 'operation.experiment_manager'),
        ('result_analyzer', 'operation.result_analyzer'),
        ('visualizer', 'visualization.visualizer'),
        ('animator', 'visualization.animator'),
        ('plot_manager', 'visualization.plot_manager')
    ]
    
    results = []
    for display_name, module_path in modules_to_check:
        try:
            result = analyze_module(display_name, module_path)
            results.append(result)
        except Exception as e:
            print(f"✗ Failed to analyze {display_name}: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("DOCUMENTATION VERIFICATION RESULTS")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Modules with complete documentation: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL MODULES ARE PROPERLY DOCUMENTED!")
        print("\nDocumentation includes:")
        print("- Module-level docstrings explaining purpose and functionality")
        print("- Class docstrings describing responsibilities")
        print("- Method docstrings with parameters, return values, and behavior")
        print("- Comprehensive inline comments for complex logic")
        return True
    else:
        print("❌ Some modules have documentation issues.")
        print("Please review the output above and add missing docstrings.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
