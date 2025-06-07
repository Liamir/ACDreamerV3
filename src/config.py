import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import os

class DotDict(dict):
    """Dictionary subclass that supports dot notation access"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Convert nested dicts to DotDict recursively
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)
            elif isinstance(value, list):
                self[key] = [DotDict(item) if isinstance(item, dict) else item for item in value]
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")



class ConfigManager:
    """
    Simple config manager for single hierarchical YAML files with CLI overrides
    Supports both dot notation (config.training.total_timesteps) and dict access
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self._config_dict = {}
        self.config = DotDict()
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> DotDict:
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                self._config_dict = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                self._config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        # Environment variable interpolation
        self._config_dict = self._interpolate_env_vars(self._config_dict)
        
        # Convert to DotDict for dot notation access
        self.config = DotDict(self._config_dict)
        
        return self.config
    
    def _interpolate_env_vars(self, obj):
        """Replace ${VAR} with environment variables"""
        if isinstance(obj, dict):
            return {k: self._interpolate_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._interpolate_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # Simple ${VAR} replacement
            import re
            pattern = r'\$\{([^}]+)\}'
            matches = re.findall(pattern, obj)
            for match in matches:
                env_val = os.environ.get(match, f"${{{match}}}")  # Keep original if not found
                obj = obj.replace(f"${{{match}}}", env_val)
            return obj
        else:
            return obj
    
    def override_from_cli(self, cli_args: argparse.Namespace):
        """Override config values from command line arguments"""
        # Convert argparse.Namespace to dict, filtering out None values
        cli_dict = {k: v for k, v in vars(cli_args).items() if v is not None}
        
        # Apply overrides using dot notation
        for key, value in cli_dict.items():
            if '.' in key:
                # Handle nested keys like "algorithm.learning_rate"
                self._set_nested_value(self._config_dict, key, value)
            else:
                # Direct key override
                if key in self._config_dict:
                    self._config_dict[key] = value
        
        # Recreate DotDict after overrides
        self.config = DotDict(self._config_dict)
    
    def _set_nested_value(self, config_dict: Dict, key_path: str, value: Any):
        """Set nested dictionary value using dot notation"""
        keys = key_path.split('.')
        current = config_dict
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value
    
    def get(self, key_path: str, default=None):
        """Get nested value using dot notation (for backward compatibility)"""
        keys = key_path.split('.')
        current = self._config_dict
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def save_config(self, output_path: str):
        """Save current config to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self._config_dict, f, default_flow_style=False, indent=2)


    def to_dict(self) -> Dict:
        """Return the configuration as a regular dictionary"""
        return self._config_dict