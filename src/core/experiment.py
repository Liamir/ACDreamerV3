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
    
    def find_model_from_config(self, cfg):
        """Find model path based on config information"""
        # Reconstruct experiment folder pattern
        k = cfg.experiment.num_envs
        ac_str = f'{k}ac_' if k > 1 else ''
        run_name = cfg.experiment.name
        env_name = cfg.experiment.env_import.split('-')[0].lower()
        model_type = cfg.evaluation.model_type
        
        # Pattern to match experiment folders
        folder_pattern = f"{cfg.algorithm.name.lower()}_{ac_str}{env_name}_{run_name}_*"
        search_pattern = self.base_path / folder_pattern
        
        # Find matching experiment folders
        matching_folders = list(self.base_path.glob(folder_pattern))
        
        if not matching_folders:
            print(f"No experiment folders found matching pattern: {folder_pattern}")
            return None
        
        # Sort by timestamp (most recent first)
        matching_folders.sort(reverse=True)
        latest_experiment = matching_folders[0]
        
        print(f"Found experiment folder: {latest_experiment.name}")
        
        # Determine model file based on type
        models_dir = latest_experiment / "models"
        
        if model_type == "best":
            model_path = models_dir / "best_model.zip"
        elif model_type == "final":
            model_path = models_dir / "final_model.zip"
        elif model_type.startswith("checkpoint_"):
            model_path = models_dir / f"{model_type}.zip"
        else:
            print(f"Unknown model_type: {model_type}")
            return None
        
        # Check if model file exists
        if model_path.exists():
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