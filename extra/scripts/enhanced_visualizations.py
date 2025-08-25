#!/usr/bin/env python3

#!/usr/bin/env python3
"""
AXIOM Enhanced Visualization Runner

This script runs all enhanced AXIOM visualizations using UV environment.
"""

import sys
from pathlib import Path

# Add src to path
extra_dir = Path(__file__).parent
src_dir = extra_dir / "src"
sys.path.insert(0, str(src_dir))

from model_setup_analysis.enhanced_architecture_visualizer import EnhancedAXIOMVisualizer

def main():
    """Main visualization function."""
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

    print(f"\n✅ Generated {len(visualizations)} enhanced visualizations")
    print(f"📂 Files saved to: {vis_dir}")

    return visualizations

if __name__ == "__main__":
    visualizations = main()

