#!/usr/bin/env python3
"""
AXIOM Model Visualization Demo

This script demonstrates the comprehensive visualization capabilities
for all AXIOM model architectures and state spaces.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
extra_dir = Path(__file__).parent.parent.parent
src_dir = extra_dir / "src"
sys.path.insert(0, str(src_dir))

from model_setup_analysis.model_architecture_visualizer import AXIOMModelVisualizer


def create_sample_configs() -> Dict[str, Any]:
    """Create sample configurations for all AXIOM models."""
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
                'width': 160,
                'height': 210,
                'input_dim': 5,
                'slot_dim': 2,
                'num_slots': 16,
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
            {
                'width': 160,
                'height': 210,
                'input_dim': 5,
                'slot_dim': 4,
                'num_slots': 8,
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
            {
                'width': 160,
                'height': 210,
                'input_dim': 5,
                'slot_dim': 6,
                'num_slots': 4,
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


def create_sample_evolution_data() -> Dict[str, List[Dict[str, Any]]]:
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


def main():
    """Run the comprehensive visualization demo."""
    print("🚀 AXIOM Model Visualization Demo")
    print("=" * 50)

    # Initialize visualizer
    print("\n1. Initializing AXIOM Model Visualizer...")
    visualizer = AXIOMModelVisualizer()
    print("✓ Visualizer initialized successfully")

    # Load sample configurations
    print("\n2. Loading sample configurations...")
    configs = create_sample_configs()
    evolution_data = create_sample_evolution_data()
    print("✓ Sample configurations loaded")

    # Generate individual model visualizations
    print("\n3. Generating individual model architecture visualizations...")

    visualizations = {}

    # SMM Architecture
    print("   - Generating SMM architecture...")
    smm_path = visualizer.visualize_smm_architecture(
        configs['smm'], "demo_smm_architecture.png"
    )
    visualizations['smm'] = smm_path
    print(f"   ✓ SMM architecture saved to: {smm_path}")

    # HSMM Architecture
    print("   - Generating HSMM architecture...")
    hsmm_path = visualizer.visualize_hsmm_architecture(
        configs['hsmm'], "demo_hsmm_architecture.png"
    )
    visualizations['hsmm'] = hsmm_path
    print(f"   ✓ HSMM architecture saved to: {hsmm_path}")

    # IMM Architecture
    print("   - Generating IMM architecture...")
    imm_path = visualizer.visualize_imm_architecture(
        configs['imm'], "demo_imm_architecture.png"
    )
    visualizations['imm'] = imm_path
    print(f"   ✓ IMM architecture saved to: {imm_path}")

    # RMM Architecture
    print("   - Generating RMM architecture...")
    rmm_path = visualizer.visualize_rmm_architecture(
        configs['rmm'], "demo_rmm_architecture.png"
    )
    visualizations['rmm'] = rmm_path
    print(f"   ✓ RMM architecture saved to: {rmm_path}")

    # TMM Architecture
    print("   - Generating TMM architecture...")
    tmm_path = visualizer.visualize_tmm_architecture(
        configs['tmm'], "demo_tmm_architecture.png"
    )
    visualizations['tmm'] = tmm_path
    print(f"   ✓ TMM architecture saved to: {tmm_path}")

    # Hybrid Utils Architecture
    print("   - Generating Hybrid Utils architecture...")
    utils_path = visualizer.visualize_hybrid_utils_architecture(
        "demo_hybrid_utils_architecture.png"
    )
    visualizations['utils'] = utils_path
    print(f"   ✓ Hybrid Utils architecture saved to: {utils_path}")

    # Generate state space evolution visualizations
    print("\n4. Generating state space evolution visualizations...")

    for model_type in ['smm', 'hsmm', 'rmm', 'tmm']:
        print(f"   - Generating {model_type.upper()} state evolution...")
        evolution_path = visualizer.visualize_state_space_evolution(
            model_type, configs[model_type], evolution_data[model_type],
            f"demo_{model_type}_evolution.png"
        )
        visualizations[f'{model_type}_evolution'] = evolution_path
        print(f"   ✓ {model_type.upper()} evolution saved to: {evolution_path}")

    # Generate model relationships visualization
    print("\n5. Generating model relationships visualization...")
    relationships_path = visualizer.visualize_model_relationships(
        "demo_model_relationships.png"
    )
    visualizations['relationships'] = relationships_path
    print(f"✓ Model relationships saved to: {relationships_path}")

    # Generate comprehensive architecture overview
    print("\n6. Generating comprehensive architecture overview...")
    all_architectures_path = visualizer.visualize_all_model_architectures(
        configs, "demo_all_architectures.png"
    )
    visualizations['all_architectures'] = all_architectures_path
    print(f"✓ All architectures overview saved to: {all_architectures_path}")

    # Generate graphical abstract
    print("\n7. Generating graphical abstract...")
    abstract_path = visualizer.create_graphical_abstract(
        configs, "demo_graphical_abstract.png"
    )
    visualizations['abstract'] = abstract_path
    print(f"✓ Graphical abstract saved to: {abstract_path}")

    # Save visualization summary
    print("\n8. Saving visualization summary...")
    summary = {
        'total_visualizations': len(visualizations),
        'visualizations': visualizations,
        'configurations_used': configs,
        'generation_timestamp': str(Path('.').absolute())
    }

    summary_path = visualizer.output_dir / "demo_visualization_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Visualization summary saved to: {summary_path}")

    print("\n" + "=" * 50)
    print("🎉 AXIOM MODEL VISUALIZATION DEMO COMPLETE!")
    print("=" * 50)

    print(f"\n📊 Generated {len(visualizations)} visualizations:")
    for name, path in visualizations.items():
        print(f"   - {name}: {path}")

    print(f"\n📁 All visualizations saved to: {visualizer.output_dir}")
    print("\n🔍 Visualization files generated:")
    print("   - Individual model architectures (SMM, HSMM, IMM, RMM, TMM, Utils)")
    print("   - State space evolution diagrams")
    print("   - Model component relationships")
    print("   - Comprehensive architecture overview")
    print("   - Graphical abstract of entire AXIOM system")
    print("   - JSON summary of all generated files")

    print("\n💡 These visualizations demonstrate:")
    print("   • Modular visualization components that can be reused")
    print("   • Comprehensive coverage of all AXIOM model types")
    print("   • State space evolution over time")
    print("   • Inter-model relationships and dependencies")
    print("   • Overall system architecture and data flow")

    return visualizations


if __name__ == "__main__":
    try:
        visualizations = main()
        print("\n✅ Demo completed successfully!")
        print(f"📂 Check the output directory for {len(visualizations)} visualization files")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
