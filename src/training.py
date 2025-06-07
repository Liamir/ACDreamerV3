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
    ac_str = '{k}ac_' if k > 1 else ''
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
        tensorboard_log="./ppo_tensorboard/"
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
    

# Test model from checkpoint
def test_checkpoint(cfg, model_path=None, num_episodes=5, init_ranges=None):
    """Test a model from a specific checkpoint"""

    k = cfg.environment.num_envs
    save_path = cfg.experiment.save_path
    run_name = cfg.experiment.name
    checkpoint_path = os.path.join(save_path, f"{run_name}_{cfg.algorithm.name}_final_{cfg.training.total_timesteps}_steps")
    # checkpoint_path = f"./checkpoints/{RUN_NAME}_ppo_final_10000_steps.zip"

    model = load_from_checkpoint(checkpoint_path)
    if model is None:
        return
    
    # Create test environment
    env = ActionCoupledWrapper(
        env_fn=lambda render_mode=None: gym.make("CustomMountainCar-v0", render_mode="rgb_array"),
        k=k, render_mode="human",
        init_ranges=init_ranges
    )
    # env = ActionCoupledWrapper(
    #     env_fn=MountainCarEnv,
    #     k=k, render_mode="human",
    #     init_ranges=init_ranges
    # )
    print(f'Testing model from checkpoint: {checkpoint_path}')

    # Test for specified episodes
    for episode in range(num_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        total_reward = 0
        steps = 0

        print(f'Started episode number {episode + 1}')
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            grid = env.render()

            done = done or truncated
            total_reward += reward
            steps += 1
            
            if done:
                print(f"Episode {episode + 1}: {steps} steps, reward: {total_reward}")
                break


# Resume training from checkpoint
def resume_training_from_checkpoint(k, checkpoint_path, additional_steps=50000, checkpoint_freq=10000, save_path="./checkpoints/", run_name="continued"):
    """
    Resume training from a specific checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        additional_steps: Additional training steps
        checkpoint_freq: Checkpoint frequency for continued training
        run_name: Name prefix for continued training checkpoints
    """

    k = cfg.environment.num_envs
    ac_str = '{k}ac_' if k > 1 else ''
    run_name = cfg.experiment.name
    save_path = cfg.experiment.save_path
    env_import = cfg.experiment.env_import

    # Load model from checkpoint
    model = load_from_checkpoint(checkpoint_path)
    if model is None:
        return None
    
    # Extract step count from checkpoint filename
    import re
    step_match = re.search(r'_checkpoint_(\d+)_steps$', checkpoint_path)
    if step_match:
        starting_steps = int(step_match.group(1))
        print(f"Resuming from step: {starting_steps}")
    else:
        starting_steps = model.num_timesteps if hasattr(model, 'nu  m_timesteps') else 0
        print(f"Could not extract step count from filename, using model's timesteps: {starting_steps}")
    
    # Create environment
    env = ActionCoupledWrapper(env_fn=ContinuousCartPoleEnv, k=k)
    
    # Create evaluation environment for tracking progress
    eval_env = ActionCoupledWrapper(env_fn=ContinuousCartPoleEnv, k=k)
    eval_env = Monitor(eval_env)
    
    model.set_env(env)
    
    # Create checkpoint callback for continued training
    os.makedirs(save_path, exist_ok=True)
    
    # Custom checkpoint callback that accounts for starting steps
    class ContinuedCheckpointCallback(CheckpointCallback):
        def __init__(self, *args, starting_steps=0, **kwargs):
            super().__init__(*args, **kwargs)
            self.starting_steps = starting_steps
        
        def _on_step(self) -> bool:
            # Adjust the step count to continue from checkpoint
            if self.n_calls % self.save_freq == 0:
                current_step = self.num_timesteps
                path = os.path.join(self.save_path, f"{self.name_prefix}_{current_step}_steps")
                self.model.save(path)
                if self.verbose >= 2:
                    print(f"Saving model checkpoint to {path}")
            return True
    
    ac_str = '_ac' if k > 1 else ''

    checkpoint_callback = ContinuedCheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=save_path,
        name_prefix=f"{run_name}_ppo_{ac_str}checkpoint_cont",
        starting_steps=starting_steps,
        verbose=2
    )
    
    # Create evaluation callback for continued training
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=checkpoint_freq // 2,
        deterministic=True,
        render=False,
        verbose=1,
        n_eval_episodes=5
    )
    
    # Combine callbacks
    callbacks = [checkpoint_callback, eval_callback]
    
    # Continue training
    print(f"Resuming training from checkpoint for {additional_steps} additional steps...")
    print(f"Continued training checkpoints will use prefix: {run_name}")
    print(f"Next checkpoint will be saved at step: {starting_steps + checkpoint_freq}")
    
    model.learn(
        total_timesteps=additional_steps,
        callback=callbacks,
        reset_num_timesteps=False,  # Don't reset timestep counter
        progress_bar=True
    )
    
    # Save final continued model with total step count
    final_step_count = starting_steps + additional_steps
    final_path = os.path.join(save_path, f"{run_name}_ppo_final_cont_{final_step_count}_steps")
    model.save(final_path)
    print(f"Continued training completed. Final model saved to: {final_path}")
    
    return model


# List available checkpoints
def list_checkpoints(checkpoint_dir="./checkpoints/"):
    """List all available checkpoints in the directory"""
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist.")
        return []
    
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.zip') and 'checkpoint' in file:
            checkpoints.append(os.path.join(checkpoint_dir, file))
    
    checkpoints.sort()  # Sort by name (which includes timestep info)
    
    print(f"Available checkpoints in {checkpoint_dir}:")
    for i, checkpoint in enumerate(checkpoints):
        print(f"{i+1}. {checkpoint}")
    
    return checkpoints


# Load and test the model from checkpoint
def load_and_test_model(k):
    # Load the trained model from the .zip checkpoint
    model = PPO.load("ppo_continuous_cartpole.zip")  # or just "ppo_continuous_cartpole"
    print('loaded PPO model')
    # Create test environment
    env = ActionCoupledWrapper(env_fn=ContinuousCartPoleEnv, k=k, render_mode="human")
    print('initialized action-coupled env')

    # Test for a few episodes
    for episode in range(5):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        total_reward = 0
        steps = 0

        print(f'Started episode number {episode}')
        
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            grid = env.render()

            done = done or truncated
            total_reward += reward
            steps += 1
            
            if done:
                print(f"Episode {episode + 1}: {steps} steps, reward: {total_reward}")
                break

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

import argparse


def is_natural_number(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a natural number (must be > 0)")
    return ivalue

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

def option_list():
    list_checkpoints()

def option_resume(k):
    RUN_NAME = 'half_angle45_f30_1'
    # checkpoint_path = f"./checkpoints/half_stop_2_ppo_ac_cartpole_checkpoint_200000_steps"
    checkpoint_path = f"./checkpoints/{RUN_NAME}_ppo_ac_cartpole_checkpoint_200000_steps"
    resume_training_from_checkpoint(k, checkpoint_path, additional_steps=200000, checkpoint_freq=100000)

def option_test(cfg):
    # checkpoint_path = f"./checkpoints_continued/continued_ppo_cartpole_continued_final_400000_steps.zip"

    # INITIAL_MAX_ANGLE = 0.001
    # INITIAL_RAD_ANGLE = INITIAL_MAX_ANGLE * 2 * math.pi / 360
    # custom_init_ranges = {
    #     'x': (-0.0, 0.0),          # Cart position range
    #     'theta': (-INITIAL_RAD_ANGLE, INITIAL_RAD_ANGLE),      # Pole angle range (radians)
    #     'x_dot': (-0.0, 0.0),    # Cart velocity range
    #     'theta_dot': (-0.0, 0.0) # Pole angular velocity range
    # }

    test_checkpoint(cfg, num_episodes=3, init_ranges=None)

def option_load_test(k):
    load_and_test_model(k)

def option_inspect():
    inspect_checkpoint("./checkpoints/ppo_cartpole_checkpoint_50000_steps")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RL training or evaluation options.")
    parser.add_argument(
        "option",
        choices=["train", "list", "resume", "test", "load_test", "inspect"],
        help="Choose which operation to perform."
    )

    parser.add_argument(
        "config",
        type=str,
        help="Path of yaml config file."
    )

    parser.add_argument(
        "k",
        type=is_natural_number,
        nargs='?',
        help="Number of sub-envs in the action-coupled env."
    )

    args = parser.parse_args()
    
    config = ConfigManager(args.config)
    config.override_from_cli(args)
    cfg = config.config

    if args.option == "train":
        option_train(cfg)
    elif args.option == "list":
        option_list()
    elif args.option == "resume":
        option_resume(k=args.k)
    elif args.option == "test":
        option_test(cfg)
    elif args.option == "load_test":
        option_load_test(k=args.k)
    elif args.option == "inspect":
        option_inspect()