"""
Command Line Interface Module
Handles argument parsing and CLI command routing
"""

import argparse
from pathlib import Path
from ..core.config import ConfigManager


def create_argument_parser():
    """Create argument parser with intuitive CLI argument names"""
    parser = argparse.ArgumentParser(description="RL Training Pipeline")
    
    # Main command
    parser.add_argument("command", choices=["train", "test", "resume", "list"], 
                       help="Action to perform")
    
    # Config file
    parser.add_argument("--config", type=str, required=True,
                       help="Path to yaml configuration file")
    
    # Short CLI arguments for common overrides
    parser.add_argument("-k", "--num-envs", type=int, 
                       help="Number of action-coupled environments")
    
    parser.add_argument("--env", type=str,
                       help="Environment name (e.g., MountainCar-v0)")
    
    parser.add_argument("--steps", type=int, 
                       help="Total training timesteps")
    
    parser.add_argument("--run-name", type=str,
                       help="Experiment name")
    
    # Command-specific arguments
    parser.add_argument("--model-path", type=str,
                       help="Direct path to model file (for test/resume)")
    
    parser.add_argument("--model-type", type=str, choices=["best", "final"],
                       help="Model type to use (for test/resume)")
    
    parser.add_argument("--episodes", type=int,
                       help="Number of episodes for testing")
    
    parser.add_argument("--additional-steps", type=int,
                       help="Additional steps for resume training")
    
    return parser


def map_cli_args_to_config_paths(args):
    """Map short CLI argument names to config hierarchy paths"""
    
    # Define the mapping from CLI args to config paths
    arg_mapping = {
        "num_envs": "experiment.num_envs",
        "k": "experiment.num_envs",
        "env": "experiment.env_import", 
        "steps": "training.total_timesteps",
        "run_name": "experiment.name",
        "model_type": "evaluation.model_type",
        "episodes": "evaluation.episodes",
    }
    
    # Create new args namespace with mapped names
    mapped_args = argparse.Namespace()
    
    # Copy all original arguments first
    for attr_name, attr_value in vars(args).items():
        setattr(mapped_args, attr_name, attr_value)
    
    # Add mapped versions for config overrides
    for cli_arg, config_path in arg_mapping.items():
        if hasattr(args, cli_arg) and getattr(args, cli_arg) is not None:
            # Set the config path as an attribute
            setattr(mapped_args, config_path, getattr(args, cli_arg))
            print(f"CLI override: {config_path} = {getattr(args, cli_arg)}")
    
    return mapped_args


def load_config_with_cli_overrides(config_path, cli_args):
    """Load config and apply CLI overrides using short argument names"""
    
    # Load base configuration
    config_manager = ConfigManager(config_path)
    print(f"Loaded config from: {config_path}")
    
    # Map short CLI args to config paths
    mapped_args = map_cli_args_to_config_paths(cli_args)
    
    # Apply the mapped overrides to config
    config_manager.override_from_cli(mapped_args)
    
    return config_manager


def print_configuration_summary(cfg, command):
    """Print a summary of the final configuration"""
    print(f"\nFinal Configuration:")
    print(f"  Command: {command}")
    print(f"  Environments: {cfg.experiment.num_envs}")
    print(f"  Algorithm: {cfg.algorithm.name}")
    print(f"  Environment: {cfg.experiment.env_import}")
    print(f"  Total steps: {cfg.training.total_timesteps}")
    print(f"  Experiment: {cfg.experiment.name}")
    print("-" * 60)