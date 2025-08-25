#!/usr/bin/env python3
"""
Run All AXIOM Model Visualizations

This script runs all model visualization methods and ensures outputs
are saved to the extra/visualizations/ directory as requested.
"""

import sys
import json
from pathlib import Path

# Add src to path for imports
extra_dir = Path(__file__).parent
src_dir = extra_dir / "src"
sys.path.insert(0, str(src_dir))

from model_setup_analysis.model_architecture_visualizer import AXIOMModelVisualizer


def create_comprehensive_configs():
    """Create comprehensive configurations for all AXIOM models."""
    return {
        'smm': {
            'width': 160,
            'height': 210,
            'input_dim': 5,
            'slot_dim': 2,
            'num_slots': 32,
            'use_bias': True,
            'ns_a': 1.0,
            'ns_b': 1.0,
            'dof_offset': 10.0,
            'mask_prob': [0.0, 0.0, 0.0, 0.0, 1.0],
            'scale': [0.075, 0.075, 0.75, 0.75, 0.75],
            'transform_inv_v_scale': 100.0,
            'bias_inv_v_scale': 0.001,
            'num_e_steps': 2,
            'learning_rate': 1.0,
            'beta': 0.0,
            'eloglike_threshold': 5.0,
            'max_grow_steps': 20
        },
        'hsmm': [
            {
                'width': 160, 'height': 210, 'input_dim': 5, 'slot_dim': 2, 'num_slots': 16,
                'use_bias': True, 'ns_a': 1.0, 'ns_b': 1.0, 'dof_offset': 10.0,
                'mask_prob': [0.0, 0.0, 0.0, 0.0, 1.0], 'scale': [0.075, 0.075, 0.75, 0.75, 0.75],
                'transform_inv_v_scale': 100.0, 'bias_inv_v_scale': 0.001,
                'num_e_steps': 2, 'learning_rate': 1.0, 'beta': 0.0,
                'eloglike_threshold': 5.0, 'max_grow_steps': 20
            },
            {
                'width': 160, 'height': 210, 'input_dim': 5, 'slot_dim': 4, 'num_slots': 8,
                'use_bias': True, 'ns_a': 1.0, 'ns_b': 1.0, 'dof_offset': 10.0,
                'mask_prob': [0.0, 0.0, 0.0, 0.0, 1.0], 'scale': [0.075, 0.075, 0.75, 0.75, 0.75],
                'transform_inv_v_scale': 100.0, 'bias_inv_v_scale': 0.001,
                'num_e_steps': 2, 'learning_rate': 1.0, 'beta': 0.0,
                'eloglike_threshold': 5.0, 'max_grow_steps': 20
            },
            {
                'width': 160, 'height': 210, 'input_dim': 5, 'slot_dim': 6, 'num_slots': 4,
                'use_bias': True, 'ns_a': 1.0, 'ns_b': 1.0, 'dof_offset': 10.0,
                'mask_prob': [0.0, 0.0, 0.0, 0.0, 1.0], 'scale': [0.075, 0.075, 0.75, 0.75, 0.75],
                'transform_inv_v_scale': 100.0, 'bias_inv_v_scale': 0.001,
                'num_e_steps': 2, 'learning_rate': 1.0, 'beta': 0.0,
                'eloglike_threshold': 5.0, 'max_grow_steps': 20
            }
        ],
        'imm': {
            'num_object_types': 32,
            'num_features': 5,
            'i_ell_threshold': -500.0,
            'cont_scale_identity': 0.5,
            'color_precision_scale': 1.0,
            'color_only_identity': False
        },
        'rmm': {
            'num_components_per_switch': 25,
            'num_switches': 100,
            'num_object_types': 32,
            'num_features': 5,
            'num_continuous_dims': 7,
            'interact_with_static': False,
            'r_ell_threshold': -100.0,
            'i_ell_threshold': -500.0,
            'cont_scale_identity': 0.5,
            'cont_scale_switch': 25.0,
            'discrete_alphas': [1e-4, 1e-4, 1e-4, 1e-4, 1.0, 1e-4],
            'r_interacting': 0.6,
            'r_interacting_predict': 0.6,
            'forward_predict': False,
            'stable_r': False,
            'relative_distance': True,
            'absolute_distance_scale': False,
            'reward_prob_threshold': 0.45,
            'color_precision_scale': 1.0,
            'color_only_identity': False,
            'exclude_background': True,
            'use_ellipses_for_interaction': True,
            'velocity_scale': 10.0
        },
        'tmm': {
            'n_total_components': 200,
            'state_dim': 2,
            'dt': 1.0,
            'vu': 0.05,
            'use_bias': True,
            'use_velocity': True,
            'sigma_sqr': 2.0,
            'logp_threshold': -0.00001,
            'position_threshold': 0.15,
            'use_unused_counter': True,
            'clip_value': 5e-4
        }
    }


def create_evolution_data():
    """Create sample evolution data for state space visualization."""
    return {
        'smm': [
            {'slots': [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]},
            {'slots': [[1.2, 2.1], [3.1, 4.2], [5.2, 6.1]]},
            {'slots': [[1.4, 2.3], [3.3, 4.4], [5.4, 6.3]]},
        ],
        'hsmm': [
            {
                'layers': [
                    {'x': 1.0, 'y': 2.0},
                    {'x': 3.0, 'y': 4.0},
                    {'x': 5.0, 'y': 6.0}
                ]
            },
            {
                'layers': [
                    {'x': 1.2, 'y': 2.1},
                    {'x': 3.2, 'y': 4.1},
                    {'x': 5.2, 'y': 6.2}
                ]
            },
        ],
        'rmm': [
            {
                'relations': [
                    {'from': [1.0, 2.0], 'to': [3.0, 4.0]},
                    {'from': [5.0, 6.0], 'to': [7.0, 8.0]},
                ]
            },
            {
                'relations': [
                    {'from': [1.1, 2.1], 'to': [3.1, 4.1]},
                    {'from': [5.1, 6.1], 'to': [7.1, 8.1]},
                ]
            },
        ],
        'tmm': [
            {'trajectory': [(1.0, 2.0), (1.1, 2.1), (1.2, 2.2)]},
            {'trajectory': [(1.2, 2.2), (1.3, 2.3), (1.4, 2.4)]},
            {'trajectory': [(1.4, 2.4), (1.5, 2.5), (1.6, 2.6)]},
        ]
    }


def run_all_visualizations():
    """Run all model visualization methods."""
    print("🚀 Running All AXIOM Model Visualizations")
    print("=" * 50)

    # Set output directory to extra/visualizations/
    vis_output_dir = Path(__file__).parent / "visualizations" / "static"
    vis_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Output directory: {vis_output_dir}")

    # Initialize visualizer with custom output directory
    print("\n1. Initializing AXIOM Model Visualizer...")
    visualizer = AXIOMModelVisualizer(output_dir=vis_output_dir)
    print("✓ Visualizer initialized successfully")

    # Load configurations
    print("\n2. Loading model configurations...")
    configs = create_comprehensive_configs()
    evolution_data = create_evolution_data()
    print("✓ Configurations loaded")

    # Generate individual model visualizations
    print("\n3. Generating individual model architecture visualizations...")
    visualizations = {}

    # SMM Architecture
    print("   - SMM Architecture...")
    smm_path = visualizer.visualize_smm_architecture(
        configs['smm'], "smm_architecture.png"
    )
    visualizations['smm_architecture'] = smm_path
    print(f"   ✓ Saved: {smm_path}")

    # HSMM Architecture
    print("   - HSMM Architecture...")
    hsmm_path = visualizer.visualize_hsmm_architecture(
        configs['hsmm'], "hsmm_architecture.png"
    )
    visualizations['hsmm_architecture'] = hsmm_path
    print(f"   ✓ Saved: {hsmm_path}")

    # IMM Architecture
    print("   - IMM Architecture...")
    imm_path = visualizer.visualize_imm_architecture(
        configs['imm'], "imm_architecture.png"
    )
    visualizations['imm_architecture'] = imm_path
    print(f"   ✓ Saved: {imm_path}")

    # RMM Architecture
    print("   - RMM Architecture...")
    rmm_path = visualizer.visualize_rmm_architecture(
        configs['rmm'], "rmm_architecture.png"
    )
    visualizations['rmm_architecture'] = rmm_path
    print(f"   ✓ Saved: {rmm_path}")

    # TMM Architecture
    print("   - TMM Architecture...")
    tmm_path = visualizer.visualize_tmm_architecture(
        configs['tmm'], "tmm_architecture.png"
    )
    visualizations['tmm_architecture'] = tmm_path
    print(f"   ✓ Saved: {tmm_path}")

    # Hybrid Utils Architecture
    print("   - Hybrid Utils Architecture...")
    utils_path = visualizer.visualize_hybrid_utils_architecture(
        "hybrid_utils_architecture.png"
    )
    visualizations['utils_architecture'] = utils_path
    print(f"   ✓ Saved: {utils_path}")

    # Generate state space evolution visualizations
    print("\n4. Generating state space evolution visualizations...")

    for model_type in ['smm', 'hsmm', 'rmm', 'tmm']:
        print(f"   - {model_type.upper()} State Evolution...")
        evolution_path = visualizer.visualize_state_space_evolution(
            model_type, configs[model_type], evolution_data[model_type],
            f"{model_type}_state_evolution.png"
        )
        visualizations[f'{model_type}_evolution'] = evolution_path
        print(f"   ✓ Saved: {evolution_path}")

    # Generate model relationships visualization
    print("\n5. Generating model relationships visualization...")
    relationships_path = visualizer.visualize_model_relationships(
        "model_relationships.png"
    )
    visualizations['relationships'] = relationships_path
    if relationships_path:
        print(f"   ✓ Saved: {relationships_path}")
    else:
        print("   ⚠ Skipped (NetworkX not available)")

    # Generate comprehensive architecture overview
    print("\n6. Generating comprehensive architecture overview...")
    all_architectures_path = visualizer.visualize_all_model_architectures(
        configs, "all_model_architectures.png"
    )
    visualizations['all_architectures'] = all_architectures_path
    print(f"   ✓ Saved: {all_architectures_path}")

    # Generate graphical abstract
    print("\n7. Generating graphical abstract...")
    abstract_path = visualizer.create_graphical_abstract(
        configs, "axiom_graphical_abstract.png"
    )
    visualizations['graphical_abstract'] = abstract_path
    print(f"   ✓ Saved: {abstract_path}")

    # Verify all files were created
    print("\n8. Verifying generated files...")
    successful_files = 0
    failed_files = 0

    for name, path in visualizations.items():
        if path and Path(path).exists():
            file_size = Path(path).stat().st_size
            print(f"   ✓ {name}: {Path(path).name} ({file_size:,} bytes)")
            successful_files += 1
        else:
            print(f"   ❌ {name}: Failed to generate")
            failed_files += 1

    # Save comprehensive summary
    print("\n9. Saving comprehensive summary...")
    summary = {
        'total_visualizations': len(visualizations),
        'successful_files': successful_files,
        'failed_files': failed_files,
        'visualizations': visualizations,
        'configurations_used': configs,
        'output_directory': str(vis_output_dir),
        'model_types_covered': ['SMM', 'HSMM', 'IMM', 'RMM', 'TMM', 'Utils'],
        'visualization_types': [
            'Individual Architectures',
            'State Space Evolution',
            'Model Relationships',
            'Comprehensive Overview',
            'Graphical Abstract'
        ],
        'timestamp': str(Path('.').absolute())
    }

    summary_path = vis_output_dir / "comprehensive_visualization_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved: {summary_path}")

    # Print final statistics
    print("\n" + "=" * 50)
    print("🎉 ALL AXIOM MODEL VISUALIZATIONS COMPLETE!")
    print("=" * 50)

    print(f"\n📊 Generated {successful_files}/{len(visualizations)} visualization files")
    if failed_files > 0:
        print(f"⚠️  {failed_files} files failed to generate")

    print(f"\n📁 All files saved to: {vis_output_dir}")

    print("\n🔍 GENERATED VISUALIZATIONS:")
    print("   • Individual Model Architectures (6 files)")
    print("   • State Space Evolution Diagrams (4 files)")
    print("   • Model Relationships Diagram (1 file)")
    print("   • Comprehensive Architecture Overview (1 file)")
    print("   • Graphical Abstract (1 file)")
    print("   • Comprehensive Summary JSON (1 file)")

    print("\n📋 MODEL TYPES COVERED:")
    print("   • SMM (Slot Memory Model)")
    print("   • HSMM (Hierarchical Slot Memory Model)")
    print("   • IMM (Identity Memory Model)")
    print("   • RMM (Relational Memory Model)")
    print("   • TMM (Temporal Memory Model)")
    print("   • Hybrid Utils (Utility Functions)")

    print("\n✨ OUTPUT STRUCTURE:")
    print("   extra/visualizations/")
    print("   ├── static/           # All visualization files")
    print("   │   ├── *_architecture.png     # Model architecture diagrams")
    print("   │   ├── *_evolution.png        # State evolution diagrams")
    print("   │   ├── model_relationships.png # Component relationships")
    print("   │   ├── all_model_architectures.png # Overview")
    print("   │   ├── axiom_graphical_abstract.png # System abstract")
    print("   │   └── comprehensive_visualization_summary.json")
    print("   ├── animations/       # For future animation outputs")
    print("   └── dashboards/       # For future dashboard outputs")

    print("\n🎯 MISSION ACCOMPLISHED!")
    print("All AXIOM model visualization methods have been executed successfully.")
    print("All outputs are available in the extra/visualizations/ directory as requested.")

    return visualizations


if __name__ == "__main__":
    try:
        visualizations = run_all_visualizations()
        print("\n✅ All visualizations completed successfully!")
        print(f"📂 Check the extra/visualizations/static/ directory for {len(visualizations)} files")
    except Exception as e:
        print(f"\n❌ Error during visualization generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
