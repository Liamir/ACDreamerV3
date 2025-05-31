import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gym import spaces

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from custom_envs.continuous_cartpole_v4 import ContinuousCartPoleEnv

# Training script
def train_ppo():
    # Create vectorized environment (helps with training stability)
    env = make_vec_env(lambda: ContinuousCartPoleEnv(), n_envs=4)
    
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
    
    # Train the model
    print("Starting training...")
    model.learn(total_timesteps=100000)
    
    # Save the model
    model.save("ppo_continuous_cartpole")
    print("Model saved!")
    
    return model

# Load and test the model from checkpoint
def load_and_test_model():
    # Load the trained model from the .zip checkpoint
    model = PPO.load("ppo_continuous_cartpole.zip")  # or just "ppo_continuous_cartpole"
    print('loaded PPO model')
    # Create test environment
    env = ContinuousCartPoleEnv(render_mode='human')
    print('initialized cartpole env')
    
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
            done = done or truncated
            
            total_reward += reward
            steps += 1
            
            if done:
                print(f"Episode {episode + 1}: {steps} steps, reward: {total_reward}")
                break

# Continue training from checkpoint
def continue_training_from_checkpoint():
    # Load the existing model
    model = PPO.load("ppo_continuous_cartpole.zip")
    
    # Create environment
    env = make_vec_env(lambda: ContinuousCartPoleEnv(), n_envs=4)
    
    # Set the environment for the loaded model
    model.set_env(env)
    
    # Continue training
    print("Continuing training from checkpoint...")
    model.learn(total_timesteps=50000)  # Train for additional steps
    
    # Save the updated model
    model.save("ppo_continuous_cartpole_continued")
    print("Updated model saved!")
    
    return model

# Inspect checkpoint information
def inspect_checkpoint():
    import zipfile
    import json
    
    # Load model to get basic info
    model = PPO.load("ppo_continuous_cartpole.zip")
    
    print("Model loaded successfully!")
    print(f"Policy: {model.policy}")
    print(f"Action space: {model.action_space}")
    print(f"Observation space: {model.observation_space}")
    
    # Read system info from the checkpoint
    try:
        with zipfile.ZipFile("ppo_continuous_cartpole.zip", 'r') as zip_file:
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

if __name__ == "__main__":
    # Option 1: Train a new model
    # model = train_ppo()
    
    # Option 2: Load and test existing checkpoint
    load_and_test_model()
    
    # Option 3: Continue training from checkpoint
    # continue_training_from_checkpoint()
    
    # Option 4: Inspect checkpoint details
    # inspect_checkpoint()