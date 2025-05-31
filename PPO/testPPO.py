import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gym import spaces
from custom_envs.continuous_cartpole_v3 import ContinuousCartPoleEnv

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

# Test the trained model
def test_model():
    # Load the trained model
    model = PPO.load("ppo_continuous_cartpole")
    
    # Create test environment
    env = ContinuousCartPoleEnv()
    
    # Test for a few episodes
    for episode in range(5):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        total_reward = 0
        steps = 0
        
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                print(f"Episode {episode + 1}: {steps} steps, reward: {total_reward}")
                break

if __name__ == "__main__":
    # Train the model
    model = train_ppo()
    
    # Test the trained model
    test_model()