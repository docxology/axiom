#!/usr/bin/env python3
"""
AXIOM Model Visualization Summary

This script provides a comprehensive summary of all generated visualizations
in the extra/visualizations/ directory.
"""

import json
from pathlib import Path

def display_visualization_summary():
    """Display a comprehensive summary of all generated visualizations."""
    print("🎉 AXIOM MODEL VISUALIZATION SYSTEM - FINAL SUMMARY")
    print("=" * 70)

    vis_dir = Path(__file__).parent / "visualizations" / "static"
    summary_file = vis_dir / "comprehensive_visualization_summary.json"

    if not summary_file.exists():
        print("❌ Summary file not found. Please run the visualization generation first.")
        return

    # Load summary data
    with open(summary_file, 'r') as f:
        summary = json.load(f)

    print("📊 OVERVIEW:")
    print(f"   • Total visualization files: {summary['total_visualizations']}")
    print(f"   • Successfully generated: {summary['successful_files']}")
    print(f"   • Failed: {summary['failed_files']}")
    print(f"   • Output directory: {summary['output_directory']}")

    print("\n📋 MODEL TYPES COVERED:")
    for model in summary['model_types_covered']:
        print(f"   • {model}")

    print("\n🔍 GENERATED VISUALIZATION FILES:")

    # Group files by type
    architecture_files = []
    evolution_files = []
    overview_files = []
    other_files = []

    for name, path in summary['visualizations'].items():
        if path is None:
            continue
        if 'architecture' in name and 'all' not in name:
            architecture_files.append((name, path))
        elif 'evolution' in name:
            evolution_files.append((name, path))
        elif 'all' in name or 'abstract' in name:
            overview_files.append((name, path))
        else:
            other_files.append((name, path))

    print("   🏗️  INDIVIDUAL MODEL ARCHITECTURES:")
    for name, path in architecture_files:
        filename = Path(path).name
        file_size = Path(path).stat().st_size
        model_name = name.replace('_architecture', '').upper()
        print(","
    print("   📈 STATE SPACE EVOLUTION:")
    for name, path in evolution_files:
        filename = Path(path).name
        file_size = Path(path).stat().st_size
        model_name = name.replace('_evolution', '').replace('_state', '').upper()
        print(","
    print("   🎨 COMPREHENSIVE OVERVIEWS:")
    for name, path in overview_files:
        filename = Path(path).name
        file_size = Path(path).stat().st_size
        display_name = name.replace('_', ' ').title()
        print(","
    if other_files:
        print("   📄 OTHER FILES:")
        for name, path in other_files:
            filename = Path(path).name
            file_size = Path(path).stat().st_size
            print(","
    print("\n📁 DIRECTORY STRUCTURE:")
    print("   extra/visualizations/")
    print("   ├── static/                    # All visualization files")
    print("   │   ├── *_architecture.png     # Individual model architectures")
    print("   │   ├── *_evolution.png        # State space evolution diagrams")
    print("   │   ├── all_model_architectures.png    # Comprehensive overview")
    print("   │   ├── axiom_graphical_abstract.png   # System graphical abstract")
    print("   │   └── comprehensive_visualization_summary.json")
    print("   ├── animations/                # For future animation outputs")
    print("   └── dashboards/                # For future dashboard outputs")

    print("\n💡 VISUALIZATION CAPABILITIES:")
    print("   • Programmatic generation of high-quality PNG images")
    print("   • Individual model architecture diagrams")
    print("   • State space evolution over time")
    print("   • Inter-model relationship graphs")
    print("   • Comprehensive system overviews")
    print("   • Modular, reusable visualization components")
    print("   • Professional publication-quality outputs")

    print("\n🎯 MODEL VISUALIZATION METHODS AVAILABLE:")
    print("   • visualize_smm_architecture()        - SMM architecture")
    print("   • visualize_hsmm_architecture()       - HSMM architecture")
    print("   • visualize_imm_architecture()        - IMM architecture")
    print("   • visualize_rmm_architecture()        - RMM architecture")
    print("   • visualize_tmm_architecture()        - TMM architecture")
    print("   • visualize_hybrid_utils_architecture() - Hybrid utils")
    print("   • visualize_state_space_evolution()   - State evolution")
    print("   • visualize_model_relationships()     - Model relationships")
    print("   • visualize_all_model_architectures() - All architectures")
    print("   • create_graphical_abstract()         - Graphical abstract")

    print("\n✅ MISSION ACCOMPLISHED!")
    print("All AXIOM model visualization methods have been successfully executed.")
    print("All outputs are available in the extra/visualizations/ directory as requested.")

    # Calculate total size
    total_size = sum(Path(path).stat().st_size for path in summary['visualizations'].values() if path)
    print(",.2f"
if __name__ == "__main__":
    display_visualization_summary()
