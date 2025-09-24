#!/usr/bin/env python3
"""
Streamlined script for generating and saving phase portrait images with policy overlay
"""

import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend - must be before importing pyplot
import matplotlib.pyplot as plt
from pathlib import Path
from integration_utils import run_complete_analysis


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate phase portraits with policy overlay'
    )
    
    # Model input
    parser.add_argument(
        '--model-path',
        type=str,
        default="../runs/ppo_lv2populations_lv2pop_1env_2Msteps_ttp_random_init_state_9k_termination_linear_norm_20250921_210935/models/best_model",
        help='Path to trained PPO model'
    )
    
    # System parameters
    parser.add_argument(
        '--growth-rates',
        type=float,
        nargs=2,
        default=[0.035, 0.027],
        help='Growth rates [r_S, r_R]'
    )
    
    parser.add_argument(
        '--death-rates',
        type=float,
        nargs=2,
        default=[0.0, 0.0],
        help='Death rates [d_S, d_R]'
    )
    
    parser.add_argument(
        '--carrying-capacity',
        type=float,
        default=10000,
        help='Carrying capacity K'
    )
    
    parser.add_argument(
        '--drug-efficiency',
        type=float,
        default=1.5,
        help='Drug efficiency d_D'
    )
    
    # Visualization parameters
    parser.add_argument(
        '--policy-resolution',
        type=int,
        default=50,
        help='Grid resolution for policy computation (higher = better quality but slower)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='phase_portraits',
        help='Directory to save output images'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for saved images'
    )
    
    parser.add_argument(
        '--no-policy',
        action='store_true',
        help='Skip policy analysis and only create standard phase portraits'
    )
    
    return parser.parse_args()


def main():
    """Main execution function"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PHASE PORTRAIT GENERATOR")
    print("=" * 60)
    
    # Print configuration
    print("\nConfiguration:")
    print(f"  Model path: {args.model_path}")
    print(f"  Output directory: {output_dir}")
    print(f"  Growth rates: {args.growth_rates}")
    print(f"  Death rates: {args.death_rates}")
    print(f"  Carrying capacity: {args.carrying_capacity}")
    print(f"  Drug efficiency: {args.drug_efficiency}")
    print(f"  Policy resolution: {args.policy_resolution}")
    print(f"  DPI: {args.dpi}")
    
    if args.no_policy:
        print("\n⚠ Running without policy (standard phase portraits only)")
        model_path = None
    else:
        model_path = args.model_path
    
    print("\n" + "-" * 60)
    
    # Run analysis
    try:
        integrator, figures = run_complete_analysis(
            model_path=model_path,
            growth_rates=args.growth_rates,
            death_rates=args.death_rates,
            carrying_capacity=args.carrying_capacity,
            drug_efficiency=args.drug_efficiency,
            policy_resolution=args.policy_resolution,
            save_figures=False,  # We'll save manually with custom DPI
            output_dir=output_dir
        )
        
        # Save figures with custom DPI
        print("\nSaving images...")
        
        if figures is not None:
            if len(figures) == 3:  # With policy
                fig_policy, fig_off, fig_on = figures
                
                # Save policy portrait
                policy_path = output_dir / 'phase_portrait_with_policy.png'
                fig_policy.savefig(policy_path, dpi=args.dpi, bbox_inches='tight')
                print(f"  ✓ Saved: {policy_path}")
                
                # Save off treatment portrait
                off_path = output_dir / 'phase_portrait_off_treatment.png'
                fig_off.savefig(off_path, dpi=args.dpi, bbox_inches='tight')
                print(f"  ✓ Saved: {off_path}")
                
                # Save on treatment portrait
                on_path = output_dir / 'phase_portrait_on_treatment.png'
                fig_on.savefig(on_path, dpi=args.dpi, bbox_inches='tight')
                print(f"  ✓ Saved: {on_path}")
                
            elif len(figures) == 2:  # Without policy
                fig_off, fig_on = figures
                
                # Save off treatment portrait
                off_path = output_dir / 'phase_portrait_off_treatment.png'
                fig_off.savefig(off_path, dpi=args.dpi, bbox_inches='tight')
                print(f"  ✓ Saved: {off_path}")
                
                # Save on treatment portrait
                on_path = output_dir / 'phase_portrait_on_treatment.png'
                fig_on.savefig(on_path, dpi=args.dpi, bbox_inches='tight')
                print(f"  ✓ Saved: {on_path}")
        
        # Close all figures to free memory
        plt.close('all')
        
        print("\n" + "=" * 60)
        print("✓ Analysis complete!")
        print(f"✓ Images saved to: {output_dir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())