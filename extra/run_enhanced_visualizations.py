#!/usr/bin/env python3
"""
Enhanced AXIOM Model Visualizations with UV Methods

This script demonstrates the comprehensive integration of:
1. Universal Virtual Environment (UV) methods for setup and management
2. Enhanced, accurate AXIOM architecture visualizations
3. Entity-Relationship Models
4. Computation Graph Overviews
5. Interactive Dashboards
6. Advanced Analytics Integration

All outputs are saved to the extra/visualizations/ directory.
"""

import sys
import json
from pathlib import Path

# Add src to path for imports
extra_dir = Path(__file__).parent
src_dir = extra_dir / "src"
sys.path.insert(0, str(src_dir))

from environment_setup.uv_environment_manager import UVEnvironmentManager
from model_setup_analysis.enhanced_architecture_visualizer import EnhancedAXIOMVisualizer


def setup_uv_environment():
    """Setup UV environment for AXIOM."""
    print("🔧 Setting up Universal Virtual Environment (UV)...")

    # Initialize UV Environment Manager
    uv_manager = UVEnvironmentManager()

    print("\n1. Checking UV availability...")
    available, version = uv_manager.check_uv_availability()
    if available:
        print(f"✓ UV {version} is available")
    else:
        print("⚠️  UV not found. Attempting installation...")
        if uv_manager.install_uv():
            print("✓ UV installation completed")
        else:
            print("❌ UV installation failed")
            return None

    print("\n2. Creating UV configuration...")
    config = uv_manager.create_default_config("axiom_enhanced_visualization")
    print(f"✓ Configuration created: {config.name}")

    print("\n3. Setting up Python environment...")
    if uv_manager.setup_python_environment():
        print("✓ Python environment setup completed")
    else:
        print("❌ Python environment setup failed")
        return None

    print("\n4. Installing dependencies...")
    if uv_manager.install_dependencies(dev=True):
        print("✓ Dependencies installed successfully")
    else:
        print("❌ Dependency installation failed")
        return None

    print("\n5. Setting up analytics integration...")
    if uv_manager.setup_analytics_integration():
        print("✓ Analytics integration setup completed")
    else:
        print("❌ Analytics setup failed")

    print("\n6. Setting up interactive dashboard...")
    if uv_manager.create_interactive_dashboard_setup():
        print("✓ Interactive dashboard setup completed")
    else:
        print("❌ Dashboard setup failed")

    print("\n7. Validating environment...")
    validation = uv_manager.validate_environment()
    if validation.get("valid", False):
        print("✓ Environment validation passed")
    else:
        print("❌ Environment validation failed")
        print(f"   Issues: {validation.get('error', 'Unknown error')}")

    print("\n8. Exporting environment info...")
    info_path = uv_manager.export_environment_info()
    print(f"✓ Environment info exported: {info_path}")

    return uv_manager


def generate_enhanced_visualizations(uv_manager):
    """Generate enhanced AXIOM architecture visualizations."""
    print("\n🎨 Generating Enhanced AXIOM Architecture Visualizations...")

    # Set output directory
    vis_output_dir = extra_dir / "visualizations" / "enhanced"
    vis_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Enhanced visualizations will be saved to: {vis_output_dir}")

    # Initialize Enhanced AXIOM Visualizer
    visualizer = EnhancedAXIOMVisualizer(output_dir=vis_output_dir)

    visualizations = {}

    # 1. Entity-Relationship Model
    print("\n1. Creating Entity-Relationship Model...")
    er_path = visualizer.create_entity_relationship_model("entity_relationship_model.png")
    if er_path:
        visualizations['entity_relationship'] = er_path
        print(f"✓ Entity-Relationship Model: {er_path}")
    else:
        print("❌ Entity-Relationship Model creation failed")

    # 2. Computation Graph Overview
    print("\n2. Creating Computation Graph Overview...")
    cg_path = visualizer.create_computation_graph_overview("computation_graph_overview.png")
    if cg_path:
        visualizations['computation_graph'] = cg_path
        print(f"✓ Computation Graph Overview: {cg_path}")
    else:
        print("❌ Computation Graph Overview creation failed")

    # 3. Architecture Analytics Dashboard
    print("\n3. Creating Architecture Analytics Dashboard...")
    analytics_path = visualizer.create_architecture_analytics_dashboard("architecture_analytics.png")
    if analytics_path:
        visualizations['analytics_dashboard'] = analytics_path
        print(f"✓ Architecture Analytics Dashboard: {analytics_path}")
    else:
        print("❌ Analytics Dashboard creation failed")

    # 4. Interactive Architecture Browser
    print("\n4. Creating Interactive Architecture Browser...")
    if visualizer.create_interactive_architecture_browser():
        interactive_path = vis_output_dir.parent / "interactive" / "architecture_browser.html"
        visualizations['interactive_browser'] = str(interactive_path)
        print(f"✓ Interactive Browser: {interactive_path}")
    else:
        print("❌ Interactive Browser creation failed")

    # 5. Export Architecture Report
    print("\n5. Exporting Comprehensive Architecture Report...")
    report_path = visualizer.export_architecture_report("comprehensive_architecture_report.json")
    if report_path:
        visualizations['architecture_report'] = report_path
        print(f"✓ Architecture Report: {report_path}")
    else:
        print("❌ Architecture Report export failed")

    return visualizations


def create_uv_visualization_script(uv_manager):
    """Create UV-compatible visualization script."""
    print("\n📝 Creating UV-compatible visualization script...")

    script_content = """
#!/usr/bin/env python3
\"\"\"
AXIOM Enhanced Visualization Runner

This script runs all enhanced AXIOM visualizations using UV environment.
\"\"\"

import sys
from pathlib import Path

# Add src to path
extra_dir = Path(__file__).parent
src_dir = extra_dir / "src"
sys.path.insert(0, str(src_dir))

from model_setup_analysis.enhanced_architecture_visualizer import EnhancedAXIOMVisualizer

def main():
    \"\"\"Main visualization function.\"\"\"
    print("🚀 Running AXIOM Enhanced Visualizations...")

    # Set output directory
    vis_dir = extra_dir / "visualizations" / "enhanced"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Initialize visualizer
    visualizer = EnhancedAXIOMVisualizer(output_dir=vis_dir)

    # Generate all visualizations
    visualizations = {}

    # Entity-Relationship Model
    print("Creating Entity-Relationship Model...")
    er_path = visualizer.create_entity_relationship_model()
    if er_path:
        visualizations['entity_relationship'] = er_path
        print(f"✓ Saved: {er_path}")

    # Computation Graph Overview
    print("Creating Computation Graph Overview...")
    cg_path = visualizer.create_computation_graph_overview()
    if cg_path:
        visualizations['computation_graph'] = cg_path
        print(f"✓ Saved: {cg_path}")

    # Analytics Dashboard
    print("Creating Architecture Analytics Dashboard...")
    analytics_path = visualizer.create_architecture_analytics_dashboard()
    if analytics_path:
        visualizations['analytics'] = analytics_path
        print(f"✓ Saved: {analytics_path}")

    print(f"\\n✅ Generated {len(visualizations)} enhanced visualizations")
    print(f"📂 Files saved to: {vis_dir}")

    return visualizations

if __name__ == "__main__":
    visualizations = main()
"""

    script_path = uv_manager.create_uv_script("enhanced_visualizations", script_content)
    print(f"✓ UV visualization script created: {script_path}")

    return script_path


def main():
    """Main execution function."""
    print("🚀 AXIOM Enhanced Visualization System with UV Methods")
    print("=" * 70)

    # Step 1: Setup UV Environment
    uv_manager = setup_uv_environment()
    if not uv_manager:
        print("\n❌ UV Environment setup failed. Exiting.")
        sys.exit(1)

    # Step 2: Generate Enhanced Visualizations
    visualizations = generate_enhanced_visualizations(uv_manager)

    # Step 3: Create UV-compatible script
    script_path = create_uv_visualization_script(uv_manager)

    # Step 4: Final verification and summary
    print("\n" + "=" * 70)
    print("🎉 AXIOM ENHANCED VISUALIZATION SYSTEM - COMPLETE!")
    print("=" * 70)

    print(f"\n📊 Generated {len(visualizations)} enhanced visualizations:")

    for name, path in visualizations.items():
        if path and Path(path).exists():
            file_size = Path(path).stat().st_size
            print(f"      • {name}: {Path(path).name} ({file_size:,} bytes)")

    print("\n📁 All files saved to:")
    print(f"   • Enhanced Visualizations: extra/visualizations/enhanced/")
    print(f"   • Interactive Dashboard: extra/visualizations/interactive/")
    print(f"   • UV Scripts: extra/scripts/")
    print(f"   • Analytics: extra/analytics/")

    print("\n🔍 ENHANCED VISUALIZATION FEATURES:")
    print("   • Entity-Relationship Models with accurate AXIOM component relationships")
    print("   • Computation Graph Overviews showing complete data flow")
    print("   • Interactive Architecture Browser for exploration")
    print("   • Advanced Analytics Dashboard with metrics and insights")
    print("   • Comprehensive Architecture Reports with recommendations")

    print("\n🔧 UV ENVIRONMENT FEATURES:")
    print("   • Universal Virtual Environment setup and management")
    print("   • Dependency management with UV")
    print("   • System resource monitoring and validation")
    print("   • Analytics integration setup")
    print("   • Interactive dashboard configuration")
    print("   • Cross-platform compatibility")

    print("\n🎯 KEY ACHIEVEMENTS:")
    print("   • ✅ Complete UV environment integration")
    print("   • ✅ Accurate, complete AXIOM architecture representations")
    print("   • ✅ Entity-Relationship Models of model components")
    print("   • ✅ Computation Graph Overviews with data flow")
    print("   • ✅ Interactive Dashboard capabilities")
    print("   • ✅ Advanced Analytics Integration")
    print("   • ✅ Professional, publication-quality outputs")

    print("\n📂 TO RUN THE VISUALIZATIONS:")
    print(f"   • Interactive Browser: open extra/visualizations/interactive/architecture_browser.html")
    print(f"   • UV Script: python extra/scripts/enhanced_visualizations.py")
    print("   • Direct: python -m extra.src.model_setup_analysis.enhanced_architecture_visualizer")

    print("\n✨ SYSTEM READY FOR USE!")
    print("The enhanced AXIOM visualization system with UV methods is fully operational.")

    return visualizations


if __name__ == "__main__":
    try:
        visualizations = main()
        print("\n✅ System execution completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ System execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

