"""
Main script for running phase portrait analysis with policy overlay
"""

import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
from integration_utils import run_complete_analysis, PolicyPhasePortraitIntegrator
from policy_analyzer import LV2PopulationWithPolicy
from base_analyzer import LV2PopulationBase


def example_standard_analysis():
    """Example: Standard phase portrait analysis without policy"""
    
    print("=" * 60)
    print("STANDARD PHASE PORTRAIT ANALYSIS")
    print("=" * 60)
    
    # Create standard analyzer
    analyzer = LV2PopulationBase(
        growth_rates=[0.035, 0.027],
        death_rates=[0.0, 0.0],
        carrying_capacity=10000,
        drug_efficiency=1.5
    )
    
    # Create phase portraits for both treatment conditions
    fig_off = analyzer.create_phase_portrait(treatment_on=False)
    fig_on = analyzer.create_phase_portrait(treatment_on=True)
    
    # Save figures
    fig_off.savefig('standard_off_treatment.png', dpi=300, bbox_inches='tight')
    fig_on.savefig('standard_on_treatment.png', dpi=300, bbox_inches='tight')
    
    print("\nSaved standard phase portraits")
    
    return analyzer, fig_off, fig_on


def example_policy_analysis_direct_path():
    """Example: Phase portrait with policy using direct model path"""
    
    print("=" * 60)
    print("POLICY-ENHANCED PHASE PORTRAIT (Direct Path)")
    print("=" * 60)
    
    # Update this path to your actual model location
    # model_path = "../runs/ppo_lv2populations_lv2pop_1env_3Msteps_ttp_random_init_state_9k_termination_linear_norm_20250923_180241/models/best_model"
    model_path = "../runs/ppo_lv2populations_lv2pop_1env_3Msteps_ttp_random_init_state_9k_termination_linear_norm_det_eval_20250924_162008/models/best_model"
    
    # Run complete analysis
    integrator, figures = run_complete_analysis(
        model_path=model_path,
        growth_rates=[0.035, 0.027],
        death_rates=[0.0, 0.0],
        carrying_capacity=10000,
        drug_efficiency=1.5,
        policy_resolution=100,  # Higher = better quality but slower
        save_figures=True,
        output_dir="output_direct"
    )
    
    if integrator and integrator.analyzer.model is not None:
        print("\n✓ Successfully created policy-enhanced phase portraits")
    else:
        print("\n✗ Failed to load policy, created standard portraits instead")
    
    return integrator, figures


def example_policy_analysis_with_loaded_model():
    """Example: Phase portrait with already loaded model object"""
    
    print("=" * 60)
    print("POLICY-ENHANCED PHASE PORTRAIT (Loaded Model)")
    print("=" * 60)
    
    # First load your model
    from stable_baselines3 import PPO
    
    model_path = "../runs/ppo_lv2populations_lv2pop_1env_2Msteps_ttp_random_init_state_9k_termination_linear_norm_20250921_210935/models/best_model"
    
    try:
        model = PPO.load(model_path)
        print(f"Loaded model from: {model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None, None
    
    # Run analysis with loaded model
    integrator, figures = run_complete_analysis(
        model_object=model,
        growth_rates=[0.035, 0.027],
        death_rates=[0.0, 0.0],
        carrying_capacity=10000,
        drug_efficiency=1.5,
        policy_resolution=100,
        save_figures=True,
        output_dir="output_loaded"
    )
    
    return integrator, figures


def example_manual_analysis():
    """Example: Manual step-by-step analysis with full control"""
    
    print("=" * 60)
    print("MANUAL POLICY ANALYSIS")
    print("=" * 60)
    
    # Step 1: Create analyzer
    analyzer = LV2PopulationWithPolicy(
        growth_rates=[0.035, 0.027],
        death_rates=[0.0, 0.0],
        carrying_capacity=10000,
        drug_efficiency=1.5
    )
    
    # Step 2: Load your model
    # model_path = "../runs/ppo_lv2populations_lv2pop_1env_2Msteps_ttp_random_init_state_9k_termination_linear_norm_20250921_210935/models/best_model"
    model_path = "../runs/ppo_lv2populations_lv2pop_1env_3Msteps_ttp_random_init_state_9k_termination_linear_norm_det_eval_20250924_162008/models/best_model"

    if not analyzer.load_policy_model(model_path):
        print("Failed to load model")
        return analyzer, None
    
    # Step 3: Compute policy grid (optional - done automatically if not called)
    print("\nComputing policy grid...")
    S_grid, R_grid, action_grid, confidence_grid = analyzer.compute_policy_grid(resolution=100)
    
    # Print some statistics
    print(f"\nPolicy Statistics:")
    print(f"  Treatment coverage: {100 * np.sum(action_grid == 1) / action_grid.size:.1f}%")
    print(f"  No treatment coverage: {100 * np.sum(action_grid == 0) / action_grid.size:.1f}%")
    print(f"  Average confidence: {np.mean(confidence_grid):.3f}")
    
    # Step 4: Create visualizations
    print("\nCreating visualizations...")
    
    # Policy-enhanced portrait
    fig_policy = analyzer.create_phase_portrait_with_policy(
        figsize=(20, 6),
        policy_resolution=100  # This is ignored if grid already computed
    )
    
    # Standard portraits for comparison
    fig_off = analyzer.create_phase_portrait(treatment_on=False)
    fig_on = analyzer.create_phase_portrait(treatment_on=True)
    
    # Save all figures
    fig_policy.savefig('manual_policy_portrait.png', dpi=300, bbox_inches='tight')
    fig_off.savefig('manual_off_treatment.png', dpi=300, bbox_inches='tight')
    fig_on.savefig('manual_on_treatment.png', dpi=300, bbox_inches='tight')
    
    print("\nSaved all figures")
    
    return analyzer, (fig_policy, fig_off, fig_on)


def example_with_config():
    """Example: Using configuration object to find model"""
    
    print("=" * 60)
    print("POLICY ANALYSIS WITH CONFIG")
    print("=" * 60)
    
    # This would be your actual config loading
    # from your_config_module import load_config
    # cfg = load_config('path/to/config.yaml')
    
    # For this example, we'll create a mock config
    class MockConfig:
        class experiment:
            num_envs = 1
            # name = "lv2pop_1env_2Msteps_ttp_random_init_state_9k_termination_linear_norm"
            name = "lv2pop_1env_3Msteps_ttp_random_init_state_9k_termination_linear_norm_det_eval"
            env_import = "lv2populations-v0"
        
        class algorithm:
            name = "ppo"
        
        class evaluation:
            model_type = "best_model"
            trial_to_eval = 0
    
    cfg = MockConfig()
    
    # Create integrator and run analysis
    integrator = PolicyPhasePortraitIntegrator(base_path=Path("../runs"))
    
    success = integrator.create_analyzer_with_policy(
        cfg=cfg,
        growth_rates=[0.035, 0.027],
        death_rates=[0.0, 0.0],
        carrying_capacity=10000,
        drug_efficiency=1.5
    )
    
    if success:
        figures = integrator.generate_analysis(
            policy_resolution=100,
            save_figures=True,
            output_dir="output_config"
        )
        return integrator, figures
    else:
        print("Failed to load policy from config")
        return integrator, None


def main():
    """Main function to run examples"""
    
    import numpy as np  # Import here for manual analysis example
    
    print("\n")
    print("=" * 60)
    print("PHASE PORTRAIT ANALYSIS WITH POLICY OVERLAY")
    print("=" * 60)
    print("\nThis script creates phase portraits with policy overlay")
    print("from trained RL models and saves them as images.\n")
    
    # Configure matplotlib to not show plots
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    # Choose which example to run (uncomment the one you want)
    
    # Example 1: Standard analysis without policy
    # analyzer, fig_off, fig_on = example_standard_analysis()
    
    # Example 2: Policy analysis with direct model path
    integrator, figures = example_policy_analysis_direct_path()
    
    # Example 3: Policy analysis with loaded model object
    # integrator, figures = example_policy_analysis_with_loaded_model()
    
    # Example 4: Manual step-by-step analysis
    # analyzer, figures = example_manual_analysis()
    
    # Example 5: Using config to find model
    # integrator, figures = example_with_config()
    
    # Close all figures to free memory
    plt.close('all')
    
    print("\n" + "=" * 60)
    print("Analysis complete! Check the output directories for saved images.")
    print("=" * 60)


if __name__ == "__main__":
    main()