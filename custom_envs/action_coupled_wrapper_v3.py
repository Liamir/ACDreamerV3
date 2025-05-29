import gymnasium as gym
import numpy as np
from gymnasium import Wrapper
import matplotlib.pyplot as plt
import time
import math
import logging
import env_register

# Set up logging
logging.basicConfig(level=logging.INFO)

class ActionCoupledWrapper(Wrapper):
    def __init__(self, env_fn, k: int, seed=None, seeds=None, init_ranges=None):
        """
        Initialize the ActionCoupledWrapper with controlled randomization and initial state.
        
        Args:
            env_fn: Function that creates a gym environment
            k: Number of environments to create
            seed: Master seed for all environments. If provided and seeds is None,
                  environments will use seed, seed+1, seed+2, etc.
            seeds: List of specific seeds for each environment. If provided, overrides seed parameter.
                  Must have length k.
            init_ranges: Dictionary mapping state variable names to (min, max) tuples specifying
                        the initialization range for each variable. Example for CartPole-v1:
                        {
                            'x': (-0.05, 0.05),        # Cart position range
                            'x_dot': (-0.01, 0.01),    # Cart velocity range
                            'theta': (-0.05, 0.05),    # Pole angle range
                            'theta_dot': (-0.01, 0.01) # Pole angular velocity range
                        }
        """
        self.k = k
        self.init_ranges = init_ranges
        
        # Handle seed initialization
        self.master_seed = seed
        
        # Global RNG for the wrapper
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        
        # Set global numpy seed as well to ensure full determinism
        if seed is not None:
            np.random.seed(seed)
        
        if seeds is not None:
            if len(seeds) != k:
                raise ValueError(f"Length of seeds list ({len(seeds)}) must match k ({k})")
            self.seeds = seeds
        elif seed is not None:
            self.seeds = [seed + i for i in range(k)]
        else:
            self.seeds = None
            
        # Create environments with specific seeds
        self.envs = []
        for i in range(k):
            env = env_fn()
            
            # Seed the environment
            if self.seeds and hasattr(env, 'seed'):
                env.seed(self.seeds[i])
            
            # Make environment deterministic if it has this attribute
            if hasattr(env.unwrapped, 'deterministic'):
                env.unwrapped.deterministic = True
                
            self.envs.append(env)
            
        self.action_space = self.envs[0].action_space
        self.observation_space = gym.spaces.Tuple([env.observation_space for env in self.envs])
        self.terminated_envs = [False] * k  # Track terminated state for each environment
        
        # For logging purposes
        if self.seeds:
            logging.info(f"Initialized environments with seeds: {self.seeds}")
        if self.init_ranges:
            logging.info(f"Using custom initialization ranges: {self.init_ranges}")
            
        # Reset immediately to apply seeds and init ranges
        self.reset(seed=seed)
        
    def reset(self, seed=None, options=None, init_ranges=None, **kwargs):
        """
        Reset all environments with controlled randomization and initial state.
        
        Args:
            seed: If provided, overrides the master seed for this reset
            options: Additional options passed to environment reset
            init_ranges: If provided, overrides the default initialization ranges
        """
        self.terminated_envs = [False] * self.k  # Reset the terminated states
        
        # Update seeds if a new master seed is provided
        if seed is not None:
            self.master_seed = seed
            self.rng = np.random.RandomState(seed)
            np.random.seed(seed)  # Set global numpy seed as well
            
            if self.seeds is not None:
                self.seeds = [seed + i for i in range(self.k)]
                logging.info(f"Updated environment seeds to: {self.seeds}")
        
        # Update initialization ranges if provided
        ranges_to_use = init_ranges if init_ranges is not None else self.init_ranges
        
        # Reset environments with appropriate seeds and initial states
        observations = []
        for i, env in enumerate(self.envs):
            reset_seed = None if self.seeds is None else self.seeds[i]
            
            # Reset the environment
            if hasattr(env, 'seed') and reset_seed is not None:
                env.seed(reset_seed)
            
            # Reset and get observation
            obs = env.reset()
            
            # Handle different return formats (gymnasium vs gym)
            if isinstance(obs, tuple):
                obs = obs[0]  # New gymnasium format returns (obs, info)
            
            # Then apply custom initialization if ranges specified
            if ranges_to_use is not None:
                self._set_custom_initial_state(env, ranges_to_use)
                # Get updated observation after state modification
                obs = self._construct_obs_from_state(env)
                
            observations.append(obs)
            
        return tuple(observations), {}
        
    def _construct_obs_from_state(self, env):
        """
        Manually construct observation from environment state.
        This is environment-specific and will be used when the environment 
        doesn't have a _get_obs or get_obs method.
        
        Args:
            env: The environment to get observation from
        """
        # For your custom ContinuousCartPoleEnv, the observation is just the state
        if hasattr(env.unwrapped, 'state') and env.unwrapped.state is not None:
            return np.array(env.unwrapped.state, dtype=np.float32)
        
        # Fallback for other environments
        env_type = type(env.unwrapped).__name__
        
        if 'CartPole' in env_type:
            # CartPole observation is just the state
            return np.array(env.unwrapped.state, dtype=np.float32)
            
        elif 'Pendulum' in env_type:
            # Pendulum observation is [cos(theta), sin(theta), theta_dot]
            return np.array(env.unwrapped.state, dtype=np.float32)
            
        elif 'MountainCar' in env_type:
            # MountainCar observation is [position, velocity]
            return np.array(env.unwrapped.state, dtype=np.float32)
            
        elif 'Acrobot' in env_type:
            # Acrobot observation is the state
            return np.array(env.unwrapped.state, dtype=np.float32)
            
        else:
            logging.warning(f"No observation construction method for {env_type}, returning current observation")
            # If we don't know how to construct the observation, just return what we have
            # This will likely be wrong but at least it won't crash
            return env.observation_space.sample()
        
    def _set_custom_initial_state(self, env, ranges):
        """
        Set environment to a custom initial state sampled from specified ranges.
        This is environment-specific and needs to be adapted for different environments.
        
        Args:
            env: The environment to modify
            ranges: Dictionary with state variable names and (min, max) ranges
        """
        # Get environment type to apply appropriate initialization
        env_type = type(env.unwrapped).__name__
        
        if 'ContinuousCartPole' in env_type or 'CartPole' in env_type:
            # CartPole state: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
            state = np.zeros(4)
            
            # If state already exists, use it as base
            if hasattr(env.unwrapped, 'state') and env.unwrapped.state is not None:
                state = np.array(env.unwrapped.state)
            
            # Sample each state variable from its specified range
            if 'x' in ranges:
                state[0] = self.rng.uniform(*ranges['x'])
            if 'x_dot' in ranges:
                state[1] = self.rng.uniform(*ranges['x_dot'])
            if 'theta' in ranges:
                state[2] = self.rng.uniform(*ranges['theta'])
            if 'theta_dot' in ranges:
                state[3] = self.rng.uniform(*ranges['theta_dot'])
            
            # Set the environment's state
            env.unwrapped.state = state
            
        elif 'Pendulum' in env_type:
            # Pendulum state: [cos(theta), sin(theta), theta_dot]
            if 'theta' in ranges:
                theta = self.rng.uniform(*ranges['theta'])
                env.unwrapped.state[0] = np.cos(theta)
                env.unwrapped.state[1] = np.sin(theta)
            if 'theta_dot' in ranges:
                env.unwrapped.state[2] = self.rng.uniform(*ranges['theta_dot'])
                
        elif 'MountainCar' in env_type:
            # MountainCar state: [position, velocity]
            if 'position' in ranges:
                env.unwrapped.state[0] = self.rng.uniform(*ranges['position'])
            if 'velocity' in ranges:
                env.unwrapped.state[1] = self.rng.uniform(*ranges['velocity'])
                
        elif 'Acrobot' in env_type:
            # Acrobot state: [cos(theta1), sin(theta1), cos(theta2), sin(theta2), theta1_dot, theta2_dot]
            state = np.zeros(6)
            if 'theta1' in ranges:
                theta1 = self.rng.uniform(*ranges['theta1'])
                state[0] = np.cos(theta1)
                state[1] = np.sin(theta1)
            if 'theta2' in ranges:
                theta2 = self.rng.uniform(*ranges['theta2'])
                state[2] = np.cos(theta2)
                state[3] = np.sin(theta2)
            if 'theta1_dot' in ranges:
                state[4] = self.rng.uniform(*ranges['theta1_dot'])
            if 'theta2_dot' in ranges:
                state[5] = self.rng.uniform(*ranges['theta2_dot'])
            env.unwrapped.state = state
            
        else:
            logging.warning(f"Custom initialization not implemented for environment type: {env_type}")
            
        # After setting state, make sure to update environment's internal state
        if hasattr(env.unwrapped, 'steps_beyond_done'):
            env.unwrapped.steps_beyond_done = None

    def step(self, action):
        obs, rewards, dones, truncs, infos = [], [], [], [], []
        
        # Use the same action for all active environments
        # This is what makes it an "action coupled" wrapper
        for i, env in enumerate(self.envs):
            if not self.terminated_envs[i]:
                # Only step environments that haven't terminated yet
                with self._seed_context(i):  # Use seed context to maintain determinism
                    result = env.step(action)
                    
                    # Handle different return formats (gym vs gymnasium)
                    if len(result) == 4:
                        # Old gym format: (obs, reward, done, info)
                        o, r, d, i_info = result
                        t = False  # No truncation in old format
                    else:
                        # New gymnasium format: (obs, reward, terminated, truncated, info)
                        o, r, d, t, i_info = result
                    
                obs.append(o)
                rewards.append(r)
                dones.append(d)
                truncs.append(t)
                infos.append(i_info)
                
                # Update terminated state for this environment
                if d or t:
                    self.terminated_envs[i] = True
            else:
                # For already terminated environments, return placeholder values
                obs.append(None)  # or the last observation if you prefer
                rewards.append(0)
                dones.append(True)
                truncs.append(True)
                infos.append({})
        
        done_flags = [d or t for d, t in zip(dones, truncs)]
        # Log number of active environments
        active_envs = sum(1 for d in self.terminated_envs if not d)
        if active_envs < self.k:  # Only log when some environments are done
            logging.info(f"Active environments: {active_envs}")
        
        return tuple(obs), sum(rewards), all(done_flags), False, {"individual_rewards": rewards, "infos": infos}
    
    def _seed_context(self, env_idx):
        """Context manager to temporarily set the RNG state for an environment step"""
        class SeedContext:
            def __init__(self, wrapper, env_idx):
                self.wrapper = wrapper
                self.env_idx = env_idx
                self.old_state = None
                
            def __enter__(self):
                if self.wrapper.seeds is not None:
                    # Save current RNG state
                    self.old_state = np.random.get_state()
                    # Set RNG state based on environment seed
                    np.random.seed(self.wrapper.seeds[self.env_idx])
                return self
                
            def __exit__(self, *args):
                if self.old_state is not None:
                    # Restore previous RNG state
                    np.random.set_state(self.old_state)
        
        return SeedContext(self, env_idx)

    def render(self):
        frames = []
        for i, env in enumerate(self.envs):
            if not self.terminated_envs[i]:
                frame = env.render()
                frames.append(frame)
            else:
                # For terminated environments, use a blank/black frame
                # This assumes the first environment has been rendered at least once
                if frames:
                    h, w, c = frames[0].shape
                    frames.append(np.zeros((h, w, c), dtype=np.uint8))
                else:
                    # If no frames yet, need to get dimensions somehow
                    frame = env.render()
                    if frame is not None:
                        frames.append(np.zeros_like(frame))
        
        if not frames or any(f is None for f in frames):
            return None

        # Determine grid size (e.g., 2x2 for 4 envs, 3x3 for 9, etc.)
        rows = math.ceil(math.sqrt(self.k))
        cols = math.ceil(self.k / rows)
        h, w, c = frames[0].shape
        grid = np.zeros((rows * h, cols * w, c), dtype=frames[0].dtype)

        for idx, frame in enumerate(frames):
            r, c_ = divmod(idx, cols)
            grid[r*h:(r+1)*h, c_*w:(c_+1)*w, :] = frame

        return grid

    def close(self):
        for env in self.envs:
            env.close()


# Import your continuous CartPole environment
# Assuming it's in the same directory or properly installed
# You'll need to adjust this import based on your setup
from continuous_cartpole_v3 import ContinuousCartPoleEnv  # Adjust import as needed

# Main loop for testing
if __name__ == "__main__":
    # Example of using custom initialization ranges for continuous CartPole
    custom_init_ranges = {
        'x': (-0.1, 0.1),          # Cart position range
        'theta': (-0.2, 0.2),      # Pole angle range (radians)
        'x_dot': (-0.05, 0.05),    # Cart velocity range
        'theta_dot': (-0.05, 0.05) # Pole angular velocity range
    }
    
    # Create wrapper with custom init ranges and enable deterministic mode
    # env = ActionCoupledWrapper(
    #     lambda: ContinuousCartPoleEnv(render_mode='rgb_array'),  # Use your continuous CartPole
    #     k=4,
    #     seed=42,  # Using a seed ensures reproducibility
    #     init_ranges=custom_init_ranges
    # )

    env = ActionCoupledWrapper(
        lambda: gym.make("ContinuousCartPole-v0", render_mode="rgb_array"), 
        # lambda: gym.make("MountainCar-v0", render_mode="rgb_array", disable_env_checker=True), 
        k=4,
        seed=42,  # Using a seed ensures reproducibility
        init_ranges=custom_init_ranges
    )
    
    # First reset already happened in __init__
    obs, _ = env.reset()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    frame = env.render()
    if frame is not None:
        im = ax.imshow(frame)
        plt.axis("off")
    
    try:
        # Use fixed seed for action sampling to ensure reproducibility
        action_rng = np.random.RandomState(42)
        
        done = False
        total_reward = 0
        step_count = 0
        
        while not done:
            # Sample continuous action deterministically
            # For continuous CartPole, action is in range [-1, 1]
            action = action_rng.uniform(env.action_space.low, env.action_space.high)
            
            result = env.step(action)
            if len(result) == 4:
                # Old gym format: (obs, reward, done, info)
                obs, reward, terminated, truncated, info = result
                t = truncated  # No truncation in old format
            else:
                # New gymnasium format: (obs, reward, terminated, truncated, info)
                obs, reward, terminated, truncated, info = result
                
            total_reward += reward
            step_count += 1
            
            frame = env.render()
            if frame is not None:
                im.set_data(frame)
                plt.draw()
                plt.pause(0.01)
            
            done = terminated or truncated
            
            # Optional: add some basic control to make it more interesting
            # This is just for demonstration - you'd want proper control logic
            if step_count % 100 == 0:
                logging.info(f"Step {step_count}, Total reward: {total_reward}")
            
        logging.info(f"Episode finished after {step_count} steps with total reward {total_reward}")
        
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    
    finally:
        env.close()
        plt.close()