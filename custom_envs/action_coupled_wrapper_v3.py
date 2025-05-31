import gymnasium as gym
import numpy as np
from gymnasium import Wrapper
import matplotlib.pyplot as plt
import time
import math
import logging
from custom_envs import env_register
from custom_envs.continuous_cartpole_v4 import ContinuousCartPoleEnv  # Adjust import as needed

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.disable(logging.CRITICAL)

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
                        the initialization range for each variable.
        """
        self.k = k
        self.init_ranges = init_ranges
        self._metadata = None

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
        
        # Set up observation and action spaces
        single_obs_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        
        # Create a stacked observation space (concatenate all observations)
        if isinstance(single_obs_space, gym.spaces.Box):
            # For Box spaces, multiply the shape by k (number of environments)
            low = np.tile(single_obs_space.low, k)
            high = np.tile(single_obs_space.high, k)
            self.observation_space = gym.spaces.Box(
                low=low, 
                high=high, 
                dtype=single_obs_space.dtype
            )
        else:
            # For other space types, you might need different handling
            raise NotImplementedError(f"Stacking not implemented for {type(single_obs_space)} spaces")
        
        self.terminated_envs = [False] * k  # Track terminated state for each environment
        
        # For logging purposes
        if self.seeds:
            logging.info(f"Initialized environments with seeds: {self.seeds}")
        if self.init_ranges:
            logging.info(f"Using custom initialization ranges: {self.init_ranges}")
            
        # Reset immediately to apply seeds and init ranges
        self.reset(seed=seed)
        
    @property
    def env(self):
        """Property to satisfy SB3's expectation of a .env attribute."""
        return self.envs[0] if self.envs else None
    
    @env.setter
    def env(self, value):
        """Setter for .env property (for SB3 compatibility)."""
        if self.envs:
            self.envs[0] = value
        
    def reset(self, seed=None, options=None, init_ranges=None, **kwargs):
        """Reset all environments and return stacked observation."""
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
        
        # Stack all observations into a single vector
        stacked_obs = np.concatenate(observations)
        return stacked_obs, {}
        
    def step(self, action):
        obs, rewards, dones, truncs, infos = [], [], [], [], []
        
        # Use the same action for all active environments
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
                # For already terminated environments, return zero observation
                zero_obs = np.zeros_like(self.envs[i].observation_space.sample())
                obs.append(zero_obs)
                rewards.append(0)
                dones.append(True)
                truncs.append(True)
                infos.append({})
        
        done_flags = [d or t for d, t in zip(dones, truncs)]
        # Log number of active environments
        active_envs = sum(1 for d in self.terminated_envs if not d)
        if active_envs < self.k:  # Only log when some environments are done
            logging.info(f"Active environments: {active_envs}")
        
        # Stack observations into a single vector
        stacked_obs = np.concatenate(obs)
        
        return stacked_obs, sum(rewards), all(done_flags), False, {"individual_rewards": rewards, "infos": infos}
    
    # ... (rest of the methods remain the same)
    def _construct_obs_from_state(self, env):
        """Manually construct observation from environment state."""
        # For your custom ContinuousCartPoleEnv, the observation is just the state
        if hasattr(env.unwrapped, 'state') and env.unwrapped.state is not None:
            return np.array(env.unwrapped.state, dtype=np.float32)
        
        # Fallback for other environments
        env_type = type(env.unwrapped).__name__
        
        if 'CartPole' in env_type:
            return np.array(env.unwrapped.state, dtype=np.float32)
        elif 'Pendulum' in env_type:
            return np.array(env.unwrapped.state, dtype=np.float32)
        elif 'MountainCar' in env_type:
            return np.array(env.unwrapped.state, dtype=np.float32)
        elif 'Acrobot' in env_type:
            return np.array(env.unwrapped.state, dtype=np.float32)
        else:
            logging.warning(f"No observation construction method for {env_type}, returning current observation")
            return env.observation_space.sample()
    
    def _set_custom_initial_state(self, env, ranges):
        """Set environment to a custom initial state sampled from specified ranges."""
        env_type = type(env.unwrapped).__name__
        
        if 'ContinuousCartPole' in env_type or 'CartPole' in env_type:
            state = np.zeros(4)
            if hasattr(env.unwrapped, 'state') and env.unwrapped.state is not None:
                state = np.array(env.unwrapped.state)
            
            if 'x' in ranges:
                state[0] = self.rng.uniform(*ranges['x'])
            if 'x_dot' in ranges:
                state[1] = self.rng.uniform(*ranges['x_dot'])
            if 'theta' in ranges:
                state[2] = self.rng.uniform(*ranges['theta'])
            if 'theta_dot' in ranges:
                state[3] = self.rng.uniform(*ranges['theta_dot'])
            
            env.unwrapped.state = state
            
        # ... (other environment types remain the same)
        
        if hasattr(env.unwrapped, 'steps_beyond_done'):
            env.unwrapped.steps_beyond_done = None

    def _seed_context(self, env_idx):
        """Context manager to temporarily set the RNG state for an environment step"""
        class SeedContext:
            def __init__(self, wrapper, env_idx):
                self.wrapper = wrapper
                self.env_idx = env_idx
                self.old_state = None
                
            def __enter__(self):
                if self.wrapper.seeds is not None:
                    self.old_state = np.random.get_state()
                    np.random.seed(self.wrapper.seeds[self.env_idx])
                return self
                
            def __exit__(self, *args):
                if self.old_state is not None:
                    np.random.set_state(self.old_state)
        
        return SeedContext(self, env_idx)

    def render(self):
        frames = []
        for i, env in enumerate(self.envs):
            if not self.terminated_envs[i]:
                frame = env.render()
                frames.append(frame)
            else:
                if frames:
                    h, w, c = frames[0].shape
                    frames.append(np.zeros((h, w, c), dtype=np.uint8))
                else:
                    frame = env.render()
                    if frame is not None:
                        frames.append(np.zeros_like(frame))
        
        if not frames or any(f is None for f in frames):
            return None

        # Create grid layout for multiple environments
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