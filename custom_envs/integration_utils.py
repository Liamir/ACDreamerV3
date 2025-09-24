"""
Integration utilities for connecting trained policies with phase portrait analysis
"""

from pathlib import Path
import matplotlib.pyplot as plt
from policy_analyzer import LV2PopulationWithPolicy


class PolicyPhasePortraitIntegrator:
    """
    Integration class for connecting trained RL policies with phase portrait analysis
    """
    
    def __init__(self, base_path=None):
        """
        Initialize integrator
        
        Parameters:
        - base_path: Base path where experiments are stored
        """
        self.base_path = Path(base_path) if base_path else Path(".")
        self.analyzer = None
        
    def find_model_from_config(self, cfg):
        """
        Find model path based on config information, including support for tuning trials
        
        Parameters:
        - cfg: Configuration object with experiment details
        
        Returns:
        - Path to model file if found, None otherwise
        """
        k = cfg.experiment.num_envs
        ac_str = f'{k}ac_' if k > 1 else ''
        run_name = cfg.experiment.name
        env_name = cfg.experiment.env_import.split('-')[0].lower()
        model_type = cfg.evaluation.model_type
        trial_to_eval = cfg.evaluation.trial_to_eval
        
        # Pattern to match experiment folders
        folder_pattern = f"{cfg.algorithm.name.lower()}_{ac_str}{env_name}_{run_name}_*"
        
        # Find matching experiment folders
        matching_folders = list(self.base_path.glob(folder_pattern))
        
        if not matching_folders:
            print(f"No experiment folders found matching pattern: {folder_pattern}")
            return None
        
        # Sort by timestamp (most recent first)
        matching_folders.sort(reverse=True)
        latest_experiment = matching_folders[0]
        
        # Look for model in the experiment folder
        model_path = latest_experiment / "models" / model_type
        
        if model_path.exists():
            return model_path
        else:
            print(f"Model not found at: {model_path}")
            return None

    def load_model(self, model_path):
        """
        Load PPO model from checkpoint
        
        Parameters:
        - model_path: Path to saved model
        
        Returns:
        - Loaded model or None if loading fails
        """
        try:
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
            print(f"Successfully loaded PPO model from: {model_path}")
            return model
        except Exception as e:
            print(f"Error loading PPO model: {e}")
            return None
    
    def create_analyzer(self, growth_rates=[0.035, 0.027], 
                       death_rates=[0.0, 0.0], 
                       carrying_capacity=10000, 
                       drug_efficiency=1.5):
        """
        Create phase portrait analyzer
        
        Parameters:
        - growth_rates: [r_S, r_R] growth rates
        - death_rates: [d_S, d_R] death rates
        - carrying_capacity: K (shared carrying capacity)
        - drug_efficiency: d_D (drug efficiency)
        
        Returns:
        - Created analyzer instance
        """
        
        self.analyzer = LV2PopulationWithPolicy(
            growth_rates=growth_rates,
            death_rates=death_rates,
            carrying_capacity=carrying_capacity,
            drug_efficiency=drug_efficiency
        )
        
        return self.analyzer
            
    def create_analyzer_with_policy(self, cfg=None, model_path=None, model_object=None,
                                   growth_rates=[0.035, 0.027], 
                                   death_rates=[0.0, 0.0], 
                                   carrying_capacity=10000, 
                                   drug_efficiency=1.5):
        """
        Create phase portrait analyzer with loaded policy
        
        Parameters:
        - cfg: Configuration object (optional)
        - model_path: Direct path to model (optional)
        - model_object: Already loaded model object (optional)
        - growth_rates, death_rates, carrying_capacity, drug_efficiency: Model parameters
        
        Returns:
        - True if successful, False otherwise
        """
        
        # Create the analyzer
        self.create_analyzer(growth_rates, death_rates, carrying_capacity, drug_efficiency)
        
        # Load the model based on what was provided
        model = None
        
        if model_object is not None:
            model = model_object
        elif model_path is not None:
            model = self.load_model(model_path)
        elif cfg is not None:
            found_path = self.find_model_from_config(cfg)
            if found_path:
                model = self.load_model(found_path)
        
        # Load the model into the analyzer
        if model is not None:
            self.analyzer.load_policy_model(model)
            print("Policy successfully integrated with phase portrait analyzer")
            return True
        else:
            print("Could not load policy model")
            return False
    
    def generate_analysis(self, policy_resolution=40, save_figures=True, output_dir="."):
        """
        Generate complete analysis with policy overlay
        
        Parameters:
        - policy_resolution: Grid resolution for policy computation
        - save_figures: Whether to save figures to disk
        - output_dir: Directory to save figures to
        
        Returns:
        - Tuple of generated figures
        """
        
        if self.analyzer is None:
            print("No analyzer created. Call create_analyzer_with_policy first.")
            return None
            
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        if self.analyzer.model is None:
            print("No policy loaded. Creating standard phase portraits...")
            fig_off = self.analyzer.create_phase_portrait(treatment_on=False)
            fig_on = self.analyzer.create_phase_portrait(treatment_on=True)
            
            if save_figures:
                fig_off.savefig(output_dir / 'phase_portrait_off_treatment.png', 
                              dpi=300, bbox_inches='tight')
                fig_on.savefig(output_dir / 'phase_portrait_on_treatment.png', 
                             dpi=300, bbox_inches='tight')
                print(f"Saved standard phase portraits to {output_dir}")
            
            return fig_off, fig_on
        else:
            print("Creating phase portrait with policy overlay...")
            fig_policy = self.analyzer.create_phase_portrait_with_policy(
                figsize=(20, 6),
                policy_resolution=policy_resolution
            )
            
            # Also create standard ones for comparison
            print("\nCreating standard phase portraits for comparison...")
            fig_off = self.analyzer.create_phase_portrait(treatment_on=False)
            fig_on = self.analyzer.create_phase_portrait(treatment_on=True)
            
            if save_figures:
                fig_policy.savefig(output_dir / 'phase_portrait_with_policy.png', 
                                 dpi=300, bbox_inches='tight')
                fig_off.savefig(output_dir / 'phase_portrait_off_treatment.png', 
                              dpi=300, bbox_inches='tight')
                fig_on.savefig(output_dir / 'phase_portrait_on_treatment.png', 
                             dpi=300, bbox_inches='tight')
                print(f"Saved all phase portrait figures to {output_dir}")
            
            return fig_policy, fig_off, fig_on


def run_complete_analysis(model_path=None, model_object=None, 
                         growth_rates=[0.035, 0.027],
                         death_rates=[0.0, 0.0],
                         carrying_capacity=10000,
                         drug_efficiency=1.5,
                         policy_resolution=40,
                         save_figures=True,
                         output_dir="."):
    """
    Convenience function to run complete analysis
    
    Parameters:
    - model_path: Path to saved PPO model
    - model_object: Already loaded PPO model object
    - growth_rates, death_rates, carrying_capacity, drug_efficiency: Model parameters
    - policy_resolution: Grid resolution for policy computation
    - save_figures: Whether to save figures
    - output_dir: Directory to save figures to
    
    Returns:
    - integrator: The integrator object with loaded analyzer
    - figures: Tuple of generated figures
    """
    
    integrator = PolicyPhasePortraitIntegrator()
    
    # Create analyzer with policy
    success = integrator.create_analyzer_with_policy(
        model_path=model_path,
        model_object=model_object,
        growth_rates=growth_rates,
        death_rates=death_rates,
        carrying_capacity=carrying_capacity,
        drug_efficiency=drug_efficiency
    )
    
    if success:
        # Generate analysis
        figures = integrator.generate_analysis(
            policy_resolution=policy_resolution,
            save_figures=save_figures,
            output_dir=output_dir
        )
        
        return integrator, figures
    else:
        print("Analysis without policy...")
        integrator.create_analyzer(growth_rates, death_rates, 
                                 carrying_capacity, drug_efficiency)
        figures = integrator.generate_analysis(
            save_figures=save_figures,
            output_dir=output_dir
        )
        
        return integrator, figures