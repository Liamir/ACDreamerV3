"""
Experiment Management Module
Handles experiment folder creation, tracking, and organization
"""

import os
import json
import glob
import yaml
from datetime import datetime
from pathlib import Path


class ExperimentManager:
    """Manages experiment lifecycle: creation, tracking, cleanup"""
    
    def __init__(self, base_path="runs"):
        self.base_path = Path(base_path)
        self.registry_path = self.base_path / "experiments.json"
        self.base_path.mkdir(exist_ok=True)
    
    def create_experiment_folder(self, cfg):
        """Create standardized experiment folder structure"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Generate experiment name
        k = cfg.experiment.num_envs
        ac_str = f'{k}ac_' if k > 1 else ''
        run_name = cfg.experiment.name
        env_name = cfg.experiment.env_import.split('-')[0].lower()
        folder_name = f"{cfg.algorithm.name.lower()}_{ac_str}{env_name}_{run_name}_{timestamp}"
        
        experiment_path = self.base_path / folder_name
        
        # Create structured subfolders
        (experiment_path / "models").mkdir(parents=True, exist_ok=True)
        (experiment_path / "logs" / "tensorboard").mkdir(parents=True, exist_ok=True)
        (experiment_path / "config").mkdir(parents=True, exist_ok=True)
        (experiment_path / "videos").mkdir(exist_ok=True)
        
        return str(experiment_path), folder_name
    
    def save_experiment_config(self, cfg, experiment_path):
        """Save the complete configuration used for this experiment"""
        config_dict = cfg.to_dict() if hasattr(cfg, 'to_dict') else cfg
        
        config_path = Path(experiment_path) / "config"
        
        # Save as both YAML (human readable) and JSON (machine readable)
        with open(config_path / "config.yaml", 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        with open(config_path / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def register_experiment(self, experiment_path, folder_name, cfg):
        """Register experiment in the global registry"""
        experiment_info = {
            "folder": folder_name,
            "path": experiment_path,
            "algorithm": cfg.algorithm.name,
            "environment": cfg.experiment.env_import,
            "total_timesteps": cfg.training.timesteps,
            "started": datetime.now().isoformat(),
            "status": "running"
        }
        
        # Load existing registry or create new
        registry = self._load_registry()
        registry[folder_name] = experiment_info
        self._save_registry(registry)
        
        return folder_name
    
    def update_experiment_status(self, experiment_id, status, additional_info=None):
        """Update experiment status in the registry"""
        registry = self._load_registry()
        
        if experiment_id not in registry:
            print(f"Warning: Experiment {experiment_id} not found in registry")
            return
        
        # Update status and completion time
        registry[experiment_id]["status"] = status
        registry[experiment_id]["updated"] = datetime.now().isoformat()
        
        if status == "completed":
            registry[experiment_id]["completed"] = datetime.now().isoformat()
        
        # Add any additional information
        if additional_info:
            registry[experiment_id].update(additional_info)
        
        self._save_registry(registry)
        print(f"Experiment {experiment_id} status updated to: {status}")
    
    def list_experiments(self):
        """List all experiments with their status"""
        if not self.base_path.exists():
            print(f"Experiments directory not found: {self.base_path}")
            return
        
        experiment_folders = [f for f in self.base_path.iterdir() 
                            if f.is_dir() and f.name != "experiments.json"]
        experiment_folders.sort(reverse=True)  # Most recent first
        
        print(f"Found {len(experiment_folders)} experiments in {self.base_path}:")
        print("-" * 80)
        
        for folder in experiment_folders:
            self._print_experiment_info(folder)


    def _find_latest_final_model(self, models_dir):
        """Find the most recent final model, including resumed ones"""
        if not models_dir.exists():
            return None
        
        # Look for all final model patterns
        final_model_patterns = [
            "final_model.zip",
            "final_model_resumed_*_steps.zip"
        ]
        
        final_models = []
        for pattern in final_model_patterns:
            final_models.extend(models_dir.glob(pattern))
        
        if not final_models:
            return None
        
        # If only one final model, return it
        if len(final_models) == 1:
            return str(final_models[0])
        
        # Multiple final models - find the one with highest step count
        best_model = None
        highest_steps = -1
        
        for model_path in final_models:
            steps = self._extract_steps_from_filename(model_path.name)
            if steps > highest_steps:
                highest_steps = steps
                best_model = model_path
        
        if best_model:
            print(f"Found multiple final models, using most recent: {best_model.name} ({highest_steps:,} steps)")
            return str(best_model)
        
        # Fallback to most recently modified file
        final_models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        print(f"Using most recently modified final model: {final_models[0].name}")
        return str(final_models[0])


    def _extract_steps_from_filename(self, filename):
        """Extract step count from filename, return 0 if not found"""
        import re
        
        # Look for patterns like "final_model_resumed_300352_steps.zip"
        step_patterns = [
            r'final_model_resumed_(\d+)_steps',
            r'final_model_(\d+)_steps',
            r'_(\d+)_steps',
        ]
        
        for pattern in step_patterns:
            match = re.search(pattern, filename)
            if match:
                return int(match.group(1))
        
        # For basic "final_model.zip", assume it's from initial training
        if filename == "final_model.zip":
            return 0
        
        return -1  # Unknown pattern

    def find_model_from_config(self, cfg):
        """Find model path based on config information, including support for tuning trials"""
        # Reconstruct experiment folder pattern
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
        
        print(f"Found experiment folder: {latest_experiment.name}")
        
        # Check if this is a tuning experiment by looking for trial folders
        trial_folders = [f for f in latest_experiment.iterdir() 
                        if f.is_dir() and f.name.startswith('trial_')]
        
        is_tuning_experiment = len(trial_folders) > 0
        
        # Handle tuning trials vs regular experiments
        if is_tuning_experiment:
            # This is a tuning experiment - look for trial subfolder
            trial_folder_name = f"trial_{trial_to_eval}"
            trial_folder_path = latest_experiment / trial_folder_name
            
            if not trial_folder_path.exists():
                print(f"Trial folder not found: {trial_folder_path}")
                self._show_available_trials(latest_experiment)
                return None
            
            print(f"Found trial folder: {trial_folder_name}")
            
            # Look for the experiment subfolder within the trial folder
            # It should contain the trial name in its folder name
            trial_subfolders = [f for f in trial_folder_path.iterdir() 
                            if f.is_dir() and trial_folder_name in f.name]
            
            if not trial_subfolders:
                print(f"No experiment subfolder found in trial folder {trial_folder_path}")
                print("Available subfolders:")
                for subfolder in trial_folder_path.iterdir():
                    if subfolder.is_dir():
                        print(f"  - {subfolder.name}")
                return None
            
            # Use the first matching subfolder (there should only be one)
            experiment_subfolder = trial_subfolders[0]
            models_dir = experiment_subfolder / "models"
            
            print(f"Using trial experiment subfolder: {experiment_subfolder.name}")
            
        else:
            # Regular experiment (not tuning)
            models_dir = latest_experiment / "models"
        
        # Determine model file based on type
        if model_type == "best":
            model_path = models_dir / "best_model.zip"
        elif model_type == "final":
            # Find the most recent final model (including resumed ones)
            model_path = self._find_latest_final_model(models_dir)
            if model_path:
                print(f"Found model: {model_path}")
                return str(model_path)
        elif model_type.startswith("checkpoint_"):
            model_path = models_dir / f"{model_type}.zip"
        else:
            print(f"Unknown model_type: {model_type}")
            return None
        
        # Check if model file exists
        if model_path and model_path.exists():
            print(f"Found model: {model_path}")
            return str(model_path)
        else:
            print(f"Model file not found: {model_path}")
            self._show_available_models(models_dir)
            return None

    def _load_registry(self):
        """Load experiment registry"""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self, registry):
        """Save experiment registry"""
        with open(self.registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
    
    def _print_experiment_info(self, folder):
        """Print information about a single experiment"""
        folder_name = folder.name
        
        # Check for models
        models_dir = folder / "models"
        available_models = []
        if models_dir.exists():
            available_models = [f.name for f in models_dir.glob("*.zip")]
        
        print(f"📁 {folder_name}")
        print(f"   Models: {', '.join(available_models) if available_models else 'None'}")
        
        # Try to read config for more info
        config_path = folder / "config" / "config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    print(f"   Algorithm: {config.get('algorithm', {}).get('name', 'Unknown')}")
                    print(f"   Environment: {config.get('experiment', {}).get('env_import', 'Unknown')}")
                    print(f"   Steps: {config.get('training', {}).get('total_timesteps', 'Unknown')}")
            except Exception:
                pass
        print()
    
    def _show_available_models(self, models_dir):
        """Show available models in a directory"""
        if models_dir.exists():
            available_models = [f.name for f in models_dir.glob("*.zip")]
            print(f"Available models in {models_dir}:")
            for model in available_models:
                print(f"  - {model}")


def print_env_info(env, cfg):
    """Print environment information"""
    print("Environment Info:")
    print(f"Env ID: {env.env_id}")
    print(f"Experiment Description: {cfg.experiment.description}")
    print(f"  Observation space: {env.observation_space}")
    print(f"    Shape: {env.observation_space.shape}")
    print(f"    Type: {type(env.observation_space)}")
    print(f"  Action space: {env.action_space}")
    print(f"    Shape: {env.action_space.shape}")
    print(f"    Type: {type(env.action_space)}")
    print(f'Training steps: {cfg.training.timesteps}')
    print()