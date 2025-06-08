import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from gym import spaces
import matplotlib.pyplot as plt
import os
import sys
import math
from datetime import datetime
import yaml
import json
import glob
import argparse

from config import ConfigManager
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import custom_envs
from custom_envs.action_coupled_wrapper_v3 import ActionCoupledWrapper
from custom_envs.continuous_cartpole_v4 import ContinuousCartPoleEnv


def print_env_info(env):
    print("Environment Info:")
    print(f"Env ID: {env.env_id}")
    print(f"Experiment Description: {cfg.experiment.description}")
    print(f"  Observation space: {env.observation_space}")
    print(f"    Shape: {env.observation_space.shape}")
    print(f"    Type: {type(env.observation_space)}")
    print(f"  Action space: {env.action_space}")
    print(f"    Shape: {env.action_space.shape}")
    print(f"    Type: {type(env.action_space)}")
    print(f'Training steps: {cfg.training.total_timesteps}')
    print()


def create_experiment_folder(cfg):
    """Create standardized experiment folder"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Simple naming: algorithm_env_timestamp
    k = cfg.environment.num_envs
    ac_str = f'{k}ac_' if k > 1 else ''
    run_name = cfg.experiment.name
    env_name = cfg.experiment.env_import.split('-')[0].lower()  # e.g., "mountaincar" from "MountainCar-v0"
    folder_name = f"{cfg.algorithm.name.lower()}_{ac_str}{env_name}_{run_name}_{timestamp}"
    
    base_path = cfg.experiment.save_path
    experiment_path = os.path.join(base_path, folder_name)
    
    # Create structured subfolders
    os.makedirs(os.path.join(experiment_path, "models"), exist_ok=True)
    os.makedirs(os.path.join(experiment_path, "logs"), exist_ok=True)
    os.makedirs(os.path.join(experiment_path, "config"), exist_ok=True)
    
    return experiment_path, folder_name


def save_experiment_config(cfg, experiment_path):
    """Save the complete configuration used for this experiment"""
    
    config_dict = cfg.to_dict() if hasattr(cfg, 'to_dict') else cfg
    
    # Save as both YAML (human readable) and JSON (machine readable)
    with open(os.path.join(experiment_path, "config", "config.yaml"), 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    with open(os.path.join(experiment_path, "config", "config.json"), 'w') as f:
        json.dump(config_dict, f, indent=2)


def register_experiment(experiment_path, folder_name, cfg):
    """Register experiment in a simple registry file"""
    registry_path = os.path.join(cfg.experiment.save_path or "runs", "experiments.json")
    
    experiment_info = {
        "folder": folder_name,
        "path": experiment_path,
        "algorithm": cfg.algorithm.name,
        "environment": cfg.experiment.env_import,
        "total_timesteps": cfg.training.total_timesteps,
        "started": datetime.now().isoformat(),
        "status": "running"
    }
    
    # Load existing registry or create new
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    else:
        registry = {}
    
    registry[folder_name] = experiment_info
    
    # Save updated registry
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    return folder_name


def update_experiment_status(cfg, experiment_id, status, additional_info=None):
    """Update experiment status in the registry"""
    registry_path = os.path.join(cfg.experiment.save_path, "experiments.json")
    
    if not os.path.exists(registry_path):
        print(f"Warning: Registry file not found at {registry_path}")
        return
    
    # Load existing registry
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
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
    
    # Save updated registry
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"Experiment {experiment_id} status updated to: {status}")


# Training script with checkpoints
def train_ppo_with_checkpoints(cfg, init_ranges=None):
    """
    Train PPO with automatic checkpoint saving
    
    Args:
        training_steps: Total training timesteps
        checkpoint_freq: Save checkpoint every N timesteps
        save_path: Directory to save checkpoints
    """

    k = cfg.environment.num_envs
    save_path = cfg.experiment.save_path
    env_import = cfg.experiment.env_import


    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    experiment_path, experiment_id = create_experiment_folder(cfg)

    register_experiment(experiment_path, experiment_id, cfg)

    save_experiment_config(cfg, experiment_path)

    print(f"Experiment ID: {experiment_id}")
    print(f"Experiment path: {experiment_path}")

    # Create vectorized environment (helps with training stability)
    env = ActionCoupledWrapper(
        env_fn=lambda render_mode=None: gym.make(env_import, render_mode="rgb_array"),
        k=k,
        init_ranges=init_ranges
    )
    
    # Create evaluation environment for tracking progress
    eval_env = ActionCoupledWrapper(
        env_fn=lambda render_mode=None: gym.make(env_import, render_mode="rgb_array"),
        k=k,
        init_ranges=init_ranges
    )

    eval_env = Monitor(eval_env)

    print_env_info(env)

    # Create experiment-specific tensorboard path
    tensorboard_path = os.path.join(experiment_path, "logs", "tensorboard")
    os.makedirs(tensorboard_path, exist_ok=True)

    # Initialize PPO model
    model = PPO(
        "MlpPolicy",  # Multi-layer perceptron policy
        env,
        verbose=1,
        learning_rate=cfg.algorithm.hyperparameters.learning_rate,
        n_steps=cfg.algorithm.hyperparameters.n_steps,      # Steps per environment per update
        batch_size=cfg.algorithm.hyperparameters.batch_size,
        n_epochs=cfg.algorithm.hyperparameters.n_epochs,
        gamma=cfg.algorithm.hyperparameters.gamma,        # Discount factor
        gae_lambda=cfg.algorithm.hyperparameters.gae_lambda,   # GAE parameter
        clip_range=cfg.algorithm.hyperparameters.clip_range,    # PPO clip range
        tensorboard_log=tensorboard_path
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.training.save_freq,
        save_path=os.path.join(experiment_path, "models"),
        name_prefix=f"checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=2
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(experiment_path, "models"),
        log_path=os.path.join(experiment_path, "logs"),
        eval_freq=cfg.training.eval_freq,
        deterministic=True,
        render=False,
        verbose=1,
        n_eval_episodes=5,
    )
    
    # Combine callbacks
    callbacks = [checkpoint_callback, eval_callback]
    
    # Train the model with checkpoints
    print(f"Starting training")
    print(f"Checkpoints will be saved to: {experiment_path}")
    print(f"TensorBoard logs: {tensorboard_path}")

    model.learn(
        total_timesteps=cfg.training.total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(experiment_path, "models", "final_model")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    update_experiment_status(cfg, experiment_id, "completed")

    return model


# Load model from specific checkpoint
def load_from_checkpoint(checkpoint_path):
    """
    Load model from a specific checkpoint
    
    Args:
        checkpoint_path: Path to the checkpoint file (without .zip extension)
    """
    try:
        model = PPO.load(checkpoint_path)
        print(f"Successfully loaded model from: {checkpoint_path}")
        return model
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def find_model_from_config(cfg, model_type="best"):
    """
    Find model path based on config information and experiment structure
    
    Args:
        cfg: Configuration object
        model_type: "best", "final", or "checkpoint_XXXX"
    
    Returns:
        str: Path to model file, or None if not found
    """
    
    # Reconstruct experiment folder pattern
    k = cfg.environment.num_envs
    ac_str = f'{k}ac_' if k > 1 else ''
    run_name = cfg.experiment.name
    env_name = cfg.experiment.env_import.split('-')[0].lower()
    base_path = cfg.experiment.save_path
    
    # Pattern to match experiment folders
    folder_pattern = f"{cfg.algorithm.name.lower()}_{ac_str}{env_name}_{run_name}_*"
    search_pattern = os.path.join(base_path, folder_pattern)
    
    # Find matching experiment folders
    matching_folders = glob.glob(search_pattern)
    
    if not matching_folders:
        print(f"No experiment folders found matching pattern: {folder_pattern}")
        return None
    
    # Sort by timestamp (most recent first)
    matching_folders.sort(reverse=True)
    latest_experiment = matching_folders[0]
    
    print(f"Found experiment folder: {os.path.basename(latest_experiment)}")
    
    # Determine model file based on type
    models_dir = os.path.join(latest_experiment, "models")
    
    if model_type == "best":
        model_path = os.path.join(models_dir, "best_model.zip")
    elif model_type == "final":
        model_path = os.path.join(models_dir, "final_model.zip")
    elif model_type.startswith("checkpoint_"):
        model_path = os.path.join(models_dir, f"{model_type}.zip")
    else:
        print(f"Unknown model_type: {model_type}")
        return None
    
    # Check if model file exists
    if os.path.exists(model_path):
        print(f"Found model: {model_path}")
        return model_path
    else:
        print(f"Model file not found: {model_path}")
        
        # Show available models in the directory
        if os.path.exists(models_dir):
            available_models = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
            print(f"Available models in {models_dir}:")
            for model in available_models:
                print(f"  - {model}")
        
        return None


def test_checkpoint(cfg, model_path=None, num_episodes=5, init_ranges=None, model_type="best"):
    """
    Test a model from a checkpoint
    
    Args:
        cfg: Configuration object
        model_path: Direct path to model file (if provided, cfg info is ignored)
        num_episodes: Number of episodes to test
        init_ranges: Initialization ranges for environment
        model_type: Type of model to load ("best", "final", or "checkpoint_XXXX")
    """
    
    k = cfg.environment.num_envs
    
    # Determine model path
    if model_path:
        # Use provided path directly
        checkpoint_path = model_path
        print(f"Using provided model path: {checkpoint_path}")
    else:
        # Find model using cfg info and experiment structure
        checkpoint_path = find_model_from_config(cfg, model_type)
        if not checkpoint_path:
            print("Could not find model. Please provide model_path or ensure experiment exists.")
            return
    
    # Load model
    model = load_from_checkpoint(checkpoint_path)
    if model is None:
        print(f"Failed to load model from: {checkpoint_path}")
        return
    
    # Create test environment
    env = ActionCoupledWrapper(
        env_fn=lambda render_mode=None: gym.make(cfg.experiment.env_import, render_mode="rgb_array"),
        k=k, render_mode="human",
        init_ranges=init_ranges
    )
    
    print(f'Testing model from checkpoint: {checkpoint_path}')
    print(f'Environment: {cfg.experiment.env_import}')
    print(f'Number of episodes: {num_episodes}')
    print("-" * 50)

    # Test for specified episodes
    episode_rewards = []
    episode_steps = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        total_reward = 0
        steps = 0

        print(f'Started episode {episode + 1}')
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            grid = env.render()

            done = done or truncated
            total_reward += reward
            steps += 1
            
            if done:
                print(f"Episode {episode + 1}: {steps} steps, reward: {total_reward}")
                episode_rewards.append(total_reward)
                episode_steps.append(steps)
                break
    
    # Print summary statistics
    print("-" * 50)
    print("TESTING SUMMARY:")
    print(f"Average reward: {sum(episode_rewards)/len(episode_rewards):.2f}")
    print(f"Average steps: {sum(episode_steps)/len(episode_steps):.2f}")
    print(f"Best reward: {max(episode_rewards):.2f}")
    print(f"Worst reward: {min(episode_rewards):.2f}")
    
    env.close()
    return episode_rewards, episode_steps


def extract_step_count(checkpoint_path, model):
    """Extract step count from checkpoint path or model"""
    import re
    
    # Try to extract from filename first
    filename = os.path.basename(checkpoint_path)
    
    # Look for patterns like "checkpoint_50000_steps" or "final_100000_steps"
    step_patterns = [
        r'checkpoint_(\d+)_steps',
        r'final_(\d+)_steps',
        r'final_model_resumed_(\d+)_steps',
        r'_(\d+)_steps'
    ]
    
    for pattern in step_patterns:
        match = re.search(pattern, filename)
        if match:
            steps = int(match.group(1))
            print(f"Extracted {steps} steps from filename: {filename}")
            return steps
    
    # Fall back to model's internal counter
    if hasattr(model, 'num_timesteps'):
        steps = model.num_timesteps
        print(f"Using model's internal timestep counter: {steps}")
        return steps
    
    # Last resort
    print("Warning: Could not determine starting step count, assuming 0")
    return 0


def find_experiment_path_from_model_path(model_path):
    """Extract experiment path from a model file path"""
    # Assume structure: experiment_folder/models/model_file.zip
    model_path = os.path.abspath(model_path)
    
    # Go up two levels: from model_file.zip -> models -> experiment_folder
    experiment_path = os.path.dirname(os.path.dirname(model_path))
    
    # Verify this looks like an experiment folder
    if os.path.exists(os.path.join(experiment_path, "config")) and \
       os.path.exists(os.path.join(experiment_path, "models")):
        return experiment_path
    else:
        print(f"Warning: {experiment_path} doesn't look like an experiment folder")
        return os.path.dirname(model_path)  # Fall back to model directory


def update_experiment_registry_for_resume(experiment_id, starting_steps, additional_steps, cfg):
    """Update experiment registry when resuming training"""
    registry_path = os.path.join(cfg.experiment.save_path, "experiments.json")
    
    if not os.path.exists(registry_path):
        print("Warning: No experiment registry found")
        return
    
    try:
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        if experiment_id in registry:
            registry[experiment_id]["status"] = "resuming"
            registry[experiment_id]["resumed_at"] = datetime.now().isoformat()
            registry[experiment_id]["resumed_from_steps"] = starting_steps
            registry[experiment_id]["additional_steps"] = additional_steps
            registry[experiment_id]["target_total_steps"] = starting_steps + additional_steps
            
            with open(registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
            
            print(f"Updated registry for resumed experiment: {experiment_id}")
        else:
            print(f"Warning: Experiment {experiment_id} not found in registry")
    
    except Exception as e:
        print(f"Warning: Could not update experiment registry: {e}")


# Resume training from checkpoint
def resume_training_from_checkpoint(cfg, model_path=None, additional_steps=None, model_type="best", init_ranges=None):
    """
    Resume training from a specific checkpoint
    
    Args:
        cfg: Configuration object
        model_path: Direct path to model file (if provided, cfg info is ignored)
        additional_steps: Additional training steps (if None, uses cfg.training.total_timesteps)
        model_type: Type of model to load ("best", "final", or "checkpoint_XXXX")
        init_ranges: Initialization ranges for environment
    
    Returns:
        model: Trained model
        experiment_path: Path to experiment folder
    """
    
    # Determine additional steps
    if additional_steps is None:
        additional_steps = cfg.training.total_timesteps
    
    # Find the model to resume from
    if model_path:
        checkpoint_path = model_path
        # Extract experiment folder from model path
        experiment_path = find_experiment_path_from_model_path(model_path)
        print(f"Using provided model path: {checkpoint_path}")
    else:
        # Find model using cfg info
        checkpoint_path = find_model_from_config(cfg, model_type)
        if not checkpoint_path:
            print("Could not find model to resume from. Please provide model_path or ensure experiment exists.")
            return None, None
        
        # Get experiment path from the found model
        experiment_path = os.path.dirname(os.path.dirname(checkpoint_path))  # Go up from models/ to experiment root
    
    print(f"Resuming training from: {checkpoint_path}")
    print(f"Experiment folder: {experiment_path}")
    
    # Load model from checkpoint
    model = load_from_checkpoint(checkpoint_path)
    if model is None:
        return None, None
    
    # Extract current step count
    starting_steps = extract_step_count(checkpoint_path, model)
    print(f"Starting from step: {starting_steps}")
    print(f"Will train for {additional_steps} additional steps")
    print(f"Final step count will be: {starting_steps + additional_steps}")
    
    # Create environments
    k = cfg.environment.num_envs
    env_import = cfg.experiment.env_import
    
    env = ActionCoupledWrapper(
        env_fn=lambda render_mode=None: gym.make(env_import, render_mode="rgb_array"),
        k=k,
        init_ranges=init_ranges
    )
    
    eval_env = ActionCoupledWrapper(
        env_fn=lambda render_mode=None: gym.make(env_import, render_mode="rgb_array"),
        k=k,
        init_ranges=init_ranges
    )
    eval_env = Monitor(eval_env)
    
    # Set environment for the loaded model
    model.set_env(env)
    
    # Update experiment status to "resuming"
    experiment_id = os.path.basename(experiment_path)
    update_experiment_registry_for_resume(experiment_id, starting_steps, additional_steps, cfg)
    
    # Create callbacks using existing experiment structure
    models_path = os.path.join(experiment_path, "models")
    logs_path = os.path.join(experiment_path, "logs")
    
    # Custom checkpoint callback that accounts for starting steps
    class ContinuedCheckpointCallback(CheckpointCallback):
        def __init__(self, *args, starting_steps=0, **kwargs):
            super().__init__(*args, **kwargs)
            self.starting_steps = starting_steps
        
        def _on_step(self) -> bool:
            if self.n_calls % self.save_freq == 0:
                # Use total timesteps (including previous training)
                total_steps = self.starting_steps + self.num_timesteps
                path = os.path.join(self.save_path, f"checkpoint_{total_steps}_steps")
                self.model.save(path)
                if self.verbose >= 2:
                    print(f"Saving resumed checkpoint to {path}")
            return True
    
    checkpoint_callback = ContinuedCheckpointCallback(
        save_freq=cfg.training.save_freq,
        save_path=models_path,
        name_prefix="checkpoint",
        starting_steps=starting_steps,
        verbose=2
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_path,
        log_path=logs_path,
        eval_freq=cfg.training.eval_freq,
        deterministic=True,
        render=False,
        verbose=1,
        n_eval_episodes=5,
    )
    
    callbacks = [checkpoint_callback, eval_callback]
    
    print("-" * 50)
    print("RESUMING TRAINING...")
    print("-" * 50)
    
    try:
        # Continue training
        model.learn(
            total_timesteps=additional_steps,
            callback=callbacks,
            reset_num_timesteps=False,  # Keep the original timestep counter
            progress_bar=True
        )
        
        # Save final model with updated step count
        final_step_count = starting_steps + additional_steps
        final_model_path = os.path.join(models_path, f"final_model_resumed_{final_step_count}_steps")
        model.save(final_model_path)
        print(f"Resumed training completed. Final model saved to: {final_model_path}")
        
        # Update experiment status
        update_experiment_status(cfg, experiment_id, "completed", {
            "resumed_from_steps": starting_steps,
            "final_steps": final_step_count,
            "additional_steps_trained": additional_steps
        })
        
    except Exception as e:
        print(f"Resume training failed: {e}")
        update_experiment_status(cfg, experiment_id, "failed", {
            "resumed_from_steps": starting_steps,
            "error": str(e),
            "resume_attempt": True
        })
        raise
    
    return model, experiment_path


def list_experiments(cfg):
    """List all experiments in the base path"""

    base_path = cfg.experiment.save_path

    if not os.path.exists(base_path):
        print(f"Experiments directory not found: {base_path}")
        return
    
    experiment_folders = glob.glob(os.path.join(base_path, "*"))
    experiment_folders = [f for f in experiment_folders if os.path.isdir(f)]
    experiment_folders.sort(reverse=True)  # Most recent first
    
    print(f"Found {len(experiment_folders)} experiments in {base_path}:")
    print("-" * 80)
    
    for folder in experiment_folders:
        folder_name = os.path.basename(folder)
        
        # Check for models
        models_dir = os.path.join(folder, "models")
        available_models = []
        if os.path.exists(models_dir):
            available_models = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
        
        print(f"📁 {folder_name}")
        print(f"   Models: {', '.join(available_models) if available_models else 'None'}")
        
        # Try to read config for more info
        config_path = os.path.join(folder, "config", "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    print(f"   Algorithm: {config.get('algorithm', {}).get('name', 'Unknown')}")
                    print(f"   Environment: {config.get('experiment', {}).get('env_import', 'Unknown')}")
                    print(f"   Steps: {config.get('training', {}).get('total_timesteps', 'Unknown')}")
            except:
                pass
        print()


# Inspect checkpoint information
def inspect_checkpoint(checkpoint_path="ppo_continuous_cartpole.zip"):
    import zipfile
    
    # Load model to get basic info
    model = PPO.load(checkpoint_path)
    
    print("Model loaded successfully!")
    print(f"Policy: {model.policy}")
    print(f"Action space: {model.action_space}")
    print(f"Observation space: {model.observation_space}")
    print(f"Number of timesteps trained: {model.num_timesteps}")
    
    # Read system info from the checkpoint
    try:
        with zipfile.ZipFile(f"{checkpoint_path}.zip" if not checkpoint_path.endswith('.zip') else checkpoint_path, 'r') as zip_file:
            if 'system_info.txt' in zip_file.namelist():
                with zip_file.open('system_info.txt') as f:
                    system_info = f.read().decode('utf-8')
                    print("\nSystem Info from checkpoint:")
                    print(system_info)
            
            if 'stable_baselines3_version' in zip_file.namelist():
                with zip_file.open('stable_baselines3_version') as f:
                    version = f.read().decode('utf-8').strip()
                    print(f"\nStable-Baselines3 version: {version}")
    except Exception as e:
        print(f"Could not read checkpoint details: {e}")


def option_train(cfg):
    # INITIAL_MAX_ANGLE = 0.001
    # INITIAL_RAD_ANGLE = INITIAL_MAX_ANGLE * 2 * math.pi / 360
    # custom_init_ranges = {
    #     'x': (-0.0, 0.0),          # Cart position range
    #     'theta': (-INITIAL_RAD_ANGLE, INITIAL_RAD_ANGLE),      # Pole angle range (radians)
    #     'x_dot': (-0.0, 0.0),    # Cart velocity range
    #     'theta_dot': (-0.0, 0.0) # Pole angular velocity range
    # }

    print(f"Starting training for {cfg.training.total_timesteps} steps, with checkpoints every {cfg.training.save_freq} steps")
    train_ppo_with_checkpoints(cfg, init_ranges=None)

def option_list(cfg):
    list_experiments(cfg)

def option_resume(cfg, model_path, additional_steps, model_type):
    resume_training_from_checkpoint(cfg, model_path, additional_steps, model_type)

def option_test(cfg, model_path, model_type, num_episodes=3):
    # checkpoint_path = f"./checkpoints_continued/continued_ppo_cartpole_continued_final_400000_steps.zip"

    # INITIAL_MAX_ANGLE = 0.001
    # INITIAL_RAD_ANGLE = INITIAL_MAX_ANGLE * 2 * math.pi / 360
    # custom_init_ranges = {
    #     'x': (-0.0, 0.0),          # Cart position range
    #     'theta': (-INITIAL_RAD_ANGLE, INITIAL_RAD_ANGLE),      # Pole angle range (radians)
    #     'x_dot': (-0.0, 0.0),    # Cart velocity range
    #     'theta_dot': (-0.0, 0.0) # Pole angular velocity range
    # }

    path = "../runs/ppo_custommountaincar_height_reward_20250607_174849/models/final_model_resumed_80000_steps.zip"
    
    test_checkpoint(cfg, model_path=model_path, model_type=model_type, num_episodes=num_episodes, init_ranges=None)


def option_inspect():
    inspect_checkpoint("./checkpoints/ppo_cartpole_checkpoint_50000_steps")


def map_cli_args_to_config_paths(args):
    """Map short CLI argument names to config hierarchy paths"""
    
    # Define the mapping from CLI args to config paths
    arg_mapping = {
        "num_envs": "environment.num_envs",
        "k": "environment.num_envs",
        "env": "experiment.env_import", 
        "steps": "training.total_timesteps",
        "run_name": "experiment.name",
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

def create_argument_parser():
    """Create argument parser with short, intuitive CLI argument names"""
    parser = argparse.ArgumentParser(description="RL Training Pipeline")
    
    # Main command
    parser.add_argument("option", choices=["train", "test", "resume", "list"], 
                       help="Action to perform")
    
    # Config file
    parser.add_argument("--config", type=str, required=True,
                       help="Path to yaml configuration file")
    
    # Short CLI arguments
    parser.add_argument("-k", "--num-envs", type=int, 
                       help="Number of action-coupled environments")
    
    parser.add_argument("--env", type=str,
                       help="Environment name (e.g., MountainCar-v0)")
    
    parser.add_argument("--steps", type=int, 
                       help="Total training timesteps")
    
    parser.add_argument("--run-name", type=str,
                       help="Experiment name")

    
    # # Action-specific arguments
    parser.add_argument("--model-path", type=str,
                       help="Direct path to model file (for test/resume)")
    
    parser.add_argument("--model-type", type=str, choices=["best", "final"], 
                       default="best",
                       help="Model type to use (for test/resume)")
    
    parser.add_argument("--episodes", type=int, default=3,
                       help="Number of episodes for testing")
    
    parser.add_argument("--additional-steps", type=int,
                       help="Additional steps for resume training")
    
    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()

    print("=" * 60)
    print("RL TRAINING PIPELINE")
    print("=" * 60)
    

    config_manager = load_config_with_cli_overrides(args.config, args)
    cfg = config_manager.config

    # config = ConfigManager(args.config)
    # config.override_from_cli(args)
    # cfg = config.config

    print(f"\nFinal Configuration:")
    print(f"  Command: {args.option}")
    print(f"  Environments: {cfg.environment.num_envs}")
    print(f"  Algorithm: {cfg.algorithm.name}")
    print(f"  Environment: {cfg.experiment.env_import}")
    print(f"  Total steps: {cfg.training.total_timesteps}")
    print(f"  Experiment: {cfg.experiment.name}")
    print("-" * 60)

    if args.option == "train":
        option_train(cfg)
    elif args.option == "list":
        option_list(cfg)
    elif args.option == "resume":
        option_resume(cfg, args.model_path, args.additional_steps, args.model_type)
    elif args.option == "test":
        option_test(cfg, args.model_path, args.model_type, args.episodes)
    elif args.option == "inspect":
        option_inspect()