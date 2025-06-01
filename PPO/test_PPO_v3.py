import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from gym import spaces
import matplotlib.pyplot as plt
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from custom_envs.continuous_cartpole_v4 import ContinuousCartPoleEnv
from custom_envs.action_coupled_wrapper_v3 import ActionCoupledWrapper

# Training script with checkpoints
def train_ppo_with_checkpoints(training_steps=100*1000, checkpoint_freq=10000, save_path="./checkpoints/", run_name="run_name"):
    """
    Train PPO with automatic checkpoint saving
    
    Args:
        training_steps: Total training timesteps
        checkpoint_freq: Save checkpoint every N timesteps
        save_path: Directory to save checkpoints
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Create vectorized environment (helps with training stability)
    env = ActionCoupledWrapper(env_fn=ContinuousCartPoleEnv, k=4)
    
    # Create evaluation environment for tracking progress
    eval_env = ActionCoupledWrapper(env_fn=ContinuousCartPoleEnv, k=4)
    eval_env = Monitor(eval_env)
    
    # Initialize PPO model
    model = PPO(
        "MlpPolicy",  # Multi-layer perceptron policy
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,      # Steps per environment per update
        batch_size=64,
        n_epochs=10,
        gamma=0.99,        # Discount factor
        gae_lambda=0.95,   # GAE parameter
        clip_range=0.2,    # PPO clip range
        tensorboard_log="./ppo_cartpole_tensorboard/"
    )
    
    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=save_path,
        name_prefix=f"{run_name}_ppo_ac_cartpole_checkpoint",  # Add run_name prefix
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=2
    )
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=checkpoint_freq // 2,
        deterministic=True,
        render=False,
        verbose=1,
        n_eval_episodes=5,
    )
    
    # Combine callbacks
    callbacks = [checkpoint_callback, eval_callback]
    
    # Train the model with checkpoints
    print(f"Starting training with checkpoints every {checkpoint_freq} steps...")
    print(f"Checkpoints will be saved to: {save_path}")
    model.learn(
        total_timesteps=training_steps,
        callback=callbacks,
        progress_bar=True  # Show progress bar
    )
    

    # Save final model
    final_model_path = os.path.join(save_path, f"{run_name}_ppo_continuous_cartpole_final")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
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

# # Resume training from checkpoint
# def resume_training_from_checkpoint(checkpoint_path, additional_steps=50000, checkpoint_freq=10000):
#     """
#     Resume training from a specific checkpoint
    
#     Args:
#         checkpoint_path: Path to checkpoint file
#         additional_steps: Additional training steps
#         checkpoint_freq: Checkpoint frequency for continued training
#     """
#     # Load model from checkpoint
#     model = load_from_checkpoint(checkpoint_path)
#     if model is None:
#         return None
    
#     # Create environment
#     env = ActionCoupledWrapper(env_fn=ContinuousCartPoleEnv, k=4)
#     model.set_env(env)
    
#     # Create checkpoint callback for continued training
#     save_path = "./checkpoints_continued/"
#     os.makedirs(save_path, exist_ok=True)
    
#     checkpoint_callback = CheckpointCallback(
#         save_freq=checkpoint_freq,
#         save_path=save_path,
#         name_prefix="ppo_cartpole_continued",
#         verbose=2
#     )
    
#     # Continue training
#     print(f"Resuming training from checkpoint for {additional_steps} additional steps...")
#     model.learn(
#         total_timesteps=additional_steps,
#         callback=checkpoint_callback,
#         reset_num_timesteps=False,  # Don't reset timestep counter
#         progress_bar=True
#     )
    
#     # Save final continued model
#     final_path = os.path.join(save_path, "ppo_cartpole_continued_final")
#     model.save(final_path)
#     print(f"Continued training completed. Final model saved to: {final_path}")
    
#     return model

# Resume training from checkpoint
def resume_training_from_checkpoint(checkpoint_path, additional_steps=50000, checkpoint_freq=10000, run_name="continued"):
    """
    Resume training from a specific checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        additional_steps: Additional training steps
        checkpoint_freq: Checkpoint frequency for continued training
        run_name: Name prefix for continued training checkpoints
    """
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
        starting_steps = model.num_timesteps if hasattr(model, 'num_timesteps') else 0
        print(f"Could not extract step count from filename, using model's timesteps: {starting_steps}")
    
    # Create environment
    env = ActionCoupledWrapper(env_fn=ContinuousCartPoleEnv, k=4)
    
    # Create evaluation environment for tracking progress
    eval_env = ActionCoupledWrapper(env_fn=ContinuousCartPoleEnv, k=4)
    eval_env = Monitor(eval_env)
    
    model.set_env(env)
    
    # Create checkpoint callback for continued training
    save_path = "./checkpoints_continued/"
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
    
    checkpoint_callback = ContinuedCheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=save_path,
        name_prefix=f"{run_name}_ppo_cartpole_continued",
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
    final_path = os.path.join(save_path, f"{run_name}_ppo_cartpole_continued_final_{final_step_count}_steps")
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

# Test model from checkpoint
def test_checkpoint(checkpoint_path, num_episodes=5):
    """Test a model from a specific checkpoint"""
    model = load_from_checkpoint(checkpoint_path)
    if model is None:
        return
    
    # Create test environment
    env = ActionCoupledWrapper(env_fn=ContinuousCartPoleEnv, k=4, render_mode="human")
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

# Load and test the model from checkpoint
def load_and_test_model():
    # Load the trained model from the .zip checkpoint
    model = PPO.load("ppo_continuous_cartpole.zip")  # or just "ppo_continuous_cartpole"
    print('loaded PPO model')
    # Create test environment
    env = ActionCoupledWrapper(env_fn=ContinuousCartPoleEnv, k=4, render_mode="human")
    print('initialized action-coupled cartpole env')

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
    import json
    
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

def option_train():
    K_TRAINING_STEPS = 200
    TOTAL_STEPS = 1000 * K_TRAINING_STEPS
    K_CHECKPOINT_FREQUENCY = 25
    CHECKPOINT_FREQUENCY = 1000 * K_CHECKPOINT_FREQUENCY
    RUN_NAME = "half_angle45_f30_1"

    print("Starting training with checkpoints...")
    model = train_ppo_with_checkpoints(
        training_steps=TOTAL_STEPS,
        checkpoint_freq=CHECKPOINT_FREQUENCY,
        save_path="./checkpoints/",
        run_name=RUN_NAME
    )

def option_list():
    list_checkpoints()

def option_resume():
    RUN_NAME = 'half_angle45_f30_1'
    # checkpoint_path = f"./checkpoints/half_stop_2_ppo_ac_cartpole_checkpoint_200000_steps"
    checkpoint_path = f"./checkpoints/{RUN_NAME}_ppo_ac_cartpole_checkpoint_100000_steps"
    resume_training_from_checkpoint(checkpoint_path, additional_steps=200000, checkpoint_freq=100000)

def option_test():
    RUN_NAME = 'half_angle45_f30_1'
    checkpoint_path = f"./checkpoints/{RUN_NAME}_ppo_continuous_cartpole_final.zip"
    # checkpoint_path = f"./checkpoints_continued/continued_ppo_cartpole_continued_final_400000_steps.zip"
    test_checkpoint(checkpoint_path, num_episodes=3)

def option_load_test():
    load_and_test_model()

def option_inspect():
    inspect_checkpoint("./checkpoints/ppo_cartpole_checkpoint_50000_steps")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RL training or evaluation options.")
    parser.add_argument(
        "option",
        choices=["train", "list", "resume", "test", "load_test", "inspect"],
        help="Choose which operation to perform."
    )
    args = parser.parse_args()

    if args.option == "train":
        option_train()
    elif args.option == "list":
        option_list()
    elif args.option == "resume":
        option_resume()
    elif args.option == "test":
        option_test()
    elif args.option == "load_test":
        option_load_test()
    elif args.option == "inspect":
        option_inspect()