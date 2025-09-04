import gymnasium as gym
import numpy as np
from gymnasium import Wrapper
import matplotlib.pyplot as plt
import time
import math
import logging
from custom_envs import env_register
from custom_envs.continuous_cartpole_v4 import ContinuousCartPoleEnv
from gym.spaces.box import Box as GymBox

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.disable(logging.CRITICAL)

class ActionCoupledWrapper(Wrapper):
    # Add metadata for render modes
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 50,
    }
    
    def __init__(self, env_fn, k: int, seed=None, seeds=None, render_mode=None, options=None, cfg=None):
        """
        Initialize the ActionCoupledWrapper with controlled randomization.
        
        Args:
            env_fn: Function that creates a gym environment
            k: Number of environments to create
            seed: Master seed for all environments. If provided and seeds is None,
                  environments will use seed, seed+1, seed+2, etc.
            seeds: List of specific seeds for each environment. If provided, overrides seed parameter.
                  Must have length k.
            render_mode: Render mode for all environments ('human', 'rgb_array', or None)
        """
        
        self.k = k
        self.original_k = k
        self._render_mode = render_mode
        self._metadata = None
        self.cfg = cfg
        self.env_fn = env_fn

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

        self.init_options = options or {}

        self.reward_type = options.get('reward_type', 'min')
        self.termination_type = options.get('termination_type', 'first')

        # Create environments with specific seeds and render mode
        self.envs = []

        for i in range(k):
            # Always create individual environments with rgb_array mode
            try:
                env = env_fn(render_mode="rgb_array")
            except TypeError:
                # Fallback if env_fn doesn't accept render_mode parameter
                env = env_fn()
                if hasattr(env, 'render_mode'):
                    env.render_mode = "rgb_array"

            # Seed the environment
            if self.seeds and hasattr(env, 'seed'):
                env.seed(self.seeds[i])
            
            # Make environment deterministic if it has this attribute
            if hasattr(env.unwrapped, 'deterministic'):
                env.unwrapped.deterministic = True
                
            self.envs.append(env)
        
        single_obs_space = self.envs[0].observation_space

        if self.cfg.stochastic_action:
            self.action_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float64)
        else:
            self.action_space = gym.spaces.Discrete(2)

        if self.cfg.agent_type == 'spatial':
            stacked_space = self._create_stacked_observation_space(single_obs_space, k)
            
            # Add 1 dimension for pop_norm global metric
            self.observation_space = gym.spaces.Box(
                low=np.concatenate([stacked_space.low, np.array([0.0])]),
                high=np.concatenate([stacked_space.high, np.array([np.inf])]),
                dtype=np.float64
            )
        elif self.cfg.agent_type == 'bulk':
            self.observation_space = gym.spaces.Box(
                low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
                high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf]),
                dtype=np.float64
            )

        self.terminated_envs = [False] * k  # Track terminated state for each environment
        
        # Rendering setup
        self._setup_rendering()
        
        # For logging purposes
        if self.seeds:
            logging.info(f"Initialized environments with seeds: {self.seeds}")
    

    def _create_stacked_observation_space(self, single_obs_space, k):
        """
        Create a stacked observation space that handles multiple space types.
        
        Args:
            single_obs_space: Observation space from a single environment
            k: Number of environments to stack
            
        Returns:
            Flattened Box space containing stacked observations
        """
        
        if isinstance(single_obs_space, gym.spaces.Box):
            # Handle Box spaces (most common case)
            low = np.tile(single_obs_space.low, k)
            high = np.tile(single_obs_space.high, k)
            return gym.spaces.Box(low=low, high=high, dtype=single_obs_space.dtype)
        
        elif isinstance(single_obs_space, gym.spaces.Dict):
            # Handle Dict spaces (like MiniGrid)
            return self._handle_dict_obs_space(single_obs_space, k)
        
        elif isinstance(single_obs_space, gym.spaces.Discrete):
            # Handle Discrete spaces by converting to Box
            # Each discrete obs becomes a one-hot vector
            n_values = single_obs_space.n
            total_size = k * n_values
            return gym.spaces.Box(
                low=0, high=1, shape=(total_size,), dtype=np.float64
            )
        
        elif isinstance(single_obs_space, gym.spaces.MultiDiscrete):
            # Handle MultiDiscrete spaces
            total_size = k * len(single_obs_space.nvec)
            max_val = np.max(single_obs_space.nvec)
            return gym.spaces.Box(
                low=0, high=max_val-1, shape=(total_size,), dtype=np.int64
            )
        
        elif isinstance(single_obs_space, gym.spaces.Tuple):
            # Handle Tuple spaces by flattening each component
            flattened_size = 0
            for space in single_obs_space.spaces:
                flattened_size += self._get_flattened_size(space)
            
            total_size = k * flattened_size
            return gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(total_size,), dtype=np.float64
            )
        
        else:
            # Fallback: try to flatten whatever we get
            try:
                sample = single_obs_space.sample()
                flattened_sample = self._flatten_observation(sample)
                total_size = k * len(flattened_sample)
                return gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(total_size,), dtype=np.float64
                )
            except:
                raise NotImplementedError(
                    f"Stacking not implemented for {type(single_obs_space)} spaces. "
                    f"Consider adding support for this space type."
                )
            

    def _handle_dict_obs_space(self, dict_space, k):
        """Handle Dictionary observation spaces (like MiniGrid)"""
        
        # Strategy 2: Flatten and concatenate all dict values
        total_size = 0
        for key, space in dict_space.spaces.items():
            if key == 'mission':  # Skip text-based mission space
                continue
            try:
                space_size = self._get_flattened_size(space)
                total_size += space_size
            except:
                print(f"  Skipping dict key '{key}': cannot determine size")
        
        if total_size > 0:
            final_size = k * total_size
            
            return gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(final_size,), dtype=np.float64
            )
        

    def _get_flattened_size(self, space):
        """Get the size of a flattened observation from a space"""
        if isinstance(space, gym.spaces.Box):
            return np.prod(space.shape)
        elif isinstance(space, gym.spaces.Discrete):
            return 1   # We represent discrete as a single float value (not one-hot)
        elif isinstance(space, gym.spaces.MultiDiscrete):
            return len(space.nvec)
        elif isinstance(space, gym.spaces.Tuple):
            return sum(self._get_flattened_size(s) for s in space.spaces)
        else:
            # Try sampling to determine size
            try:
                sample = space.sample()
                flattened = self._flatten_observation(sample)
                return len(flattened)
            except:
                return 64  # Reasonable default

    def _flatten_observation(self, obs):
        """Flatten any observation to a 1D numpy array"""
        if isinstance(obs, dict):
            # For dict observations, concatenate non-text values
            flattened_parts = []
            for key, value in obs.items():
                if key == 'mission':  # Skip text mission
                    continue
                try:
                    flat_part = self._flatten_observation(value)
                    flattened_parts.append(flat_part)
                except:
                    continue
            
            if flattened_parts:
                return np.concatenate(flattened_parts)
            else:
                return np.array([0.0])  # Fallback
        
        elif isinstance(obs, (list, tuple)):
            # Flatten sequences
            flattened_parts = [self._flatten_observation(item) for item in obs]
            return np.concatenate(flattened_parts)
        
        elif isinstance(obs, np.ndarray):
            return obs.flatten().astype(np.float64)
        
        elif isinstance(obs, (int, float)):
            return np.array([float(obs)])
        
        else:
            # Try to convert to numpy and flatten
            try:
                return np.array(obs).flatten().astype(np.float64)
            except:
                # Last resort: return a zero
                return np.array([0.0])

            
    
    def _sample_from_ranges(self, init_low, init_high, variable_name):
        """
        Sample a value from the specified range for a given variable.
        
        Args:
            init_low: Dict of low values
            init_high: Dict of high values  
            variable_name: Name of the variable to sample
            
        Returns:
            Sampled value or None if variable not in ranges
        """
        if init_low is None or init_high is None:
            return None
            
        if variable_name in init_low and variable_name in init_high:
            low = init_low[variable_name]
            high = init_high[variable_name]
            return self.rng.uniform(low, high)
            
        return None
    
    def _setup_rendering(self):
        """Setup rendering parameters based on the number of environments."""
        # Calculate grid layout
        self.render_rows = math.ceil(math.sqrt(self.k))
        self.render_cols = math.ceil(self.k / self.render_rows)
        
        # Rendering state
        self.screen = None
        self.clock = None
        self.is_rendering_setup = False
        
        # Get dimensions from first environment if available
        if hasattr(self.envs[0].unwrapped, 'screen_width'):
            self.single_env_width = self.envs[0].unwrapped.screen_width
            self.single_env_height = self.envs[0].unwrapped.screen_height
        else:
            # Default dimensions
            self.single_env_width = 600
            self.single_env_height = 400
        
        # Calculate total screen dimensions
        self.total_width = self.render_cols * self.single_env_width
        self.total_height = self.render_rows * self.single_env_height
        
        # Add padding between environments
        self.padding = 2
        self.total_width += (self.render_cols - 1) * self.padding
        self.total_height += (self.render_rows - 1) * self.padding
        
    def _init_pygame_display(self):
        """Initialize pygame display for human rendering."""
        if not self.is_rendering_setup and self.render_mode == "human":
            try:
                import pygame
                pygame.init()
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.total_width, self.total_height))
                pygame.display.set_caption(f"ActionCoupledWrapper - {self.k} Environments")
                self.clock = pygame.time.Clock()
                self.is_rendering_setup = True
            except ImportError:
                raise gym.error.DependencyNotInstalled(
                    'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
                )
        
    @property
    def env(self):
        """Property to satisfy SB3's expectation of a .env attribute."""
        return self.envs[0] if self.envs else None
    
    @env.setter
    def env(self, value):
        """Setter for .env property (for SB3 compatibility)."""
        if self.envs:
            self.envs[0] = value

    @property
    def render_mode(self):
        """Get the render mode."""
        return self._render_mode

    @render_mode.setter  
    def render_mode(self, value):
        """Set the render mode."""
        self._render_mode = value

    @property
    def env_id(self):
        spec = getattr(self.envs[0], "spec", None)
        return spec.id if spec else "unknown_env"
        
    def reset(self, options=None, seed=None, **kwargs):
        """
        Reset all environments with support for custom initialization ranges.
        
        Args:
            seed: Master seed for all environments
            options: Dict of options that can include:
                - init_low: Dict mapping state variables to low values
                - init_high: Dict mapping state variables to high values
                - Any other environment-specific options
        """

        self.terminated_envs = [False] * self.k
        self.step_count = 0
        
        # Update seeds if a new master seed is provided
        if seed is not None:
            self.master_seed = seed
            self.rng = np.random.RandomState(seed)
            np.random.seed(seed)
            
            if self.seeds is not None:
                self.seeds = [seed + i for i in range(self.k)]
                logging.info(f"Updated environment seeds to: {self.seeds}")
        
        reset_options = options or self.init_options
        
        # Reset environments
        observations = []
        total_counts = np.zeros(shape=(2), dtype=np.float64)
        self.total_initial_population = 0.0
        for i, env in enumerate(self.envs):
            reset_seed = None if self.seeds is None else self.seeds[i]
            
            if hasattr(env, 'seed') and reset_seed is not None:
                env.seed(reset_seed)
            
            # Reset and get observation
            try:
                subenv_options = {cell_type: counts[i] for cell_type, counts in reset_options['init_state'].items()}
                self.total_initial_population += np.sum(list(subenv_options.values()))
                result = env.reset(seed=reset_seed, options=subenv_options)
            except TypeError:
                # Fallback for environments that don't support options
                print(f'Failed to automatically set the initial state ranges. Trying to set them manually.')
                result = env.reset(seed=reset_seed)
                
                # Apply manual initialization if ranges provided
                if reset_options is not None:
                    self._apply_manual_initialization(env, reset_options)
                    result = self._get_observation_after_state_change(env)
            
            # Handle different return formats
            if isinstance(result, tuple):
                obs, info = result  # (obs, info) format
            else:
                obs = result  # Just obs
            
            total_counts += env.unwrapped.counts
            flattened_obs = self._flatten_observation(obs)
            observations.append(flattened_obs)

        total_population = total_counts.sum()
        self.pop_norm = total_population / self.total_initial_population
        total_population = np.array([total_population])
                
        # Stack all observations into a single vector
        stacked_obs = np.concatenate(observations)
        if self.cfg.agent_type == 'spatial':
            final_observation = np.concatenate([stacked_obs, total_population])

        elif self.cfg.agent_type == 'bulk':
            final_observation = np.concatenate([
                total_counts, 
                total_population, 
                total_population
            ])
        # Ensure the stacked observation matches our observation space
        expected_size = self.observation_space.shape[0]
        if len(final_observation) != expected_size:
            print(f"Warning: Stacked obs size {len(final_observation)} != expected {expected_size}")

        return final_observation, info
    
    def _apply_manual_initialization(self, env, options):
        """
        Manually apply initialization ranges for environments that don't support options.
        This is a fallback method for older environments.
        """
        env_type = type(env.unwrapped).__name__.lower()
        
        if 'cartpole' in env_type:
            if hasattr(env.unwrapped, 'state'):
                state = np.array(env.unwrapped.state)
                
                # Sample new values for specified variables
                for i, var_name in enumerate(['x', 'x_dot', 'theta', 'theta_dot']):
                    new_val = self._sample_from_ranges(options['init_low'], options['init_high'], var_name)
                    if new_val is not None:
                        state[i] = new_val
                
                env.unwrapped.state = state
                
        elif 'mountaincar' in env_type:
            if hasattr(env.unwrapped, 'state'):
                state = np.array(env.unwrapped.state)
                
                # Sample new values for MountainCar variables  
                for i, var_name in enumerate(['position', 'velocity']):
                    new_val = self._sample_from_ranges(options['init_low'], options['init_high'], var_name)
                    if new_val is not None:
                        state[i] = new_val
                
                env.unwrapped.state = state
        
        # Reset steps_beyond_done if it exists
        if hasattr(env.unwrapped, 'steps_beyond_done'):
            env.unwrapped.steps_beyond_done = None
    
    def _get_observation_after_state_change(self, env):
        """Get observation after manually changing environment state."""
        env_type = type(env.unwrapped).__name__.lower()
        
        # For most environments, the observation is just the state
        if hasattr(env.unwrapped, 'state') and env.unwrapped.state is not None:
            return np.array(env.unwrapped.state, dtype=np.float64)
        
        # Fallback
        return env.observation_space.sample()
        
    def step(self, action):
        obs, rewards, dones, truncs, infos = [], [], [], [], []
        total_counts = np.zeros(shape=(2), dtype=np.float64)

        # Growth (get next counts from the ODE, using the same action for all envs)
        if self.cfg.stochastic_action:
            treatment_prob = 1 / (1 + np.exp(-action[0]))
            n_treated = int(treatment_prob * self.k)
            if n_treated > 0:
                treated_indices = set(self.rng.choice(self.k, size=n_treated, replace=False))
            else:
                treated_indices = set()

        for i, env in enumerate(self.envs):
            if self.cfg.stochastic_action:
                binary_action = 1 if i in treated_indices else 0
            else:
                binary_action = action

            with self._seed_context(i):  # Use seed context to maintain determinism
                result = env.step(binary_action)
                
                # Handle different return formats (gym vs gymnasium)
                if len(result) == 4:
                    # Old gym format: (obs, reward, done, info)
                    o, r, d, i_info = result
                    t = False  # No truncation in old format
                else:
                    # New gymnasium format: (obs, reward, terminated, truncated, info)
                    o, r, d, t, i_info = result
            
            flattened_obs = self._flatten_observation(o)
            obs.append(flattened_obs)
            rewards.append(r)
            dones.append(d)
            truncs.append(t)
            infos.append(i_info)

            total_counts += env.unwrapped.counts

        self.step_count += 1

        # print('current k is', self.k)
        # print('observation from all envs:', obs)

        # self.pop_norm = self.population_size / self.original_population
        total_population = total_counts.sum()
        self.pop_norm = total_population / self.total_initial_population
        done = False
        if self.cfg.objective_type == 'TTP' and self.pop_norm >= 1.2:
            done = True
        
        elif self.cfg.objective_type == 'TB' and self.step_count >= self.cfg.params.max_episode_steps:
            done = True
    
        # print(f'{self.pop_norm = }')

        if self.reward_type == 'min':
            reward = np.min(rewards)
        elif self.reward_type == 'avg':
            reward = np.mean(rewards)
        elif self.reward_type == 'sum':
            reward = np.sum(rewards)
        else:
            print(f'Warning: reward type: {self.reward_type} not supported.')

        if self.cfg.agent_type == 'spatial':
            stacked_obs = np.concatenate(obs)
            final_observation = np.concatenate([stacked_obs, np.array([total_population], dtype=np.float64)])
        elif self.cfg.agent_type == 'bulk':
            final_observation = np.concatenate([
                total_counts, 
                np.array([total_population]), 
                np.array([total_population])
            ])

        return final_observation, reward, done, False, {"individual_rewards": rewards, "infos": infos}
    
    def render(self):
        """Render all environments in a grid layout."""
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization."
            )
            return None
        
        # Initialize pygame display if needed
        if self.render_mode == "human":
            self._init_pygame_display()
        
        # Collect frames from all environments
        frames = []
        for i, env in enumerate(self.envs):
            try:
                # Individual environments should already be in rgb_array mode
                frame = env.render()
                
                # Handle None frames (environment might not be ready to render)
                if frame is None:
                    if frames:  # Use dimensions from previous frames
                        h, w, c = frames[0].shape
                        frame = np.zeros((h, w, c), dtype=np.uint8)
                    else:  # Create a default black frame
                        frame = np.zeros((self.single_env_height, self.single_env_width, 3), dtype=np.uint8)
                
                # Add overlay for terminated environments (without numbering)
                if self.terminated_envs[i]:
                    frame = self._add_terminated_overlay(frame, i)
                
                frames.append(frame)
                
            except Exception as e:
                logging.warning(f"Failed to render environment {i}: {e}")
                # Create a red error frame
                if frames:
                    h, w, c = frames[0].shape
                else:
                    h, w, c = self.single_env_height, self.single_env_width, 3
                error_frame = np.full((h, w, c), [255, 0, 0], dtype=np.uint8)  # Red frame
                frames.append(error_frame)
        
        if not frames:
            return None
        
        # Create grid layout
        grid_image = self._create_grid_layout(frames)
        
        # Add title and progress bar to the final grid image
        final_image = self._add_title_and_progress_bar(grid_image)
        
        if self.render_mode == "human":
            return self._render_human(final_image)
        else:  # rgb_array
            return final_image
    
    def _create_grid_layout(self, frames):
        """Create a grid layout from individual environment frames."""
        if not frames:
            return None
        
        h, w, c = frames[0].shape
        
        # Create the grid canvas
        grid_height = self.render_rows * h + (self.render_rows - 1) * self.padding
        grid_width = self.render_cols * w + (self.render_cols - 1) * self.padding
        grid = np.zeros((grid_height, grid_width, c), dtype=frames[0].dtype)
        
        # Place each frame in the grid
        for idx, frame in enumerate(frames):
            if idx >= self.k:  # Don't exceed the number of environments
                break
                
            row = idx // self.render_cols
            col = idx % self.render_cols
            
            # Calculate position with padding
            y_start = row * (h + self.padding)
            y_end = y_start + h
            x_start = col * (w + self.padding)
            x_end = x_start + w
            
            grid[y_start:y_end, x_start:x_end, :] = frame
        
        return grid
    
    def _add_terminated_overlay(self, frame, env_idx):
        """Add a semi-transparent overlay to indicate terminated environment."""
        overlay = frame.copy()
        # Add red tint to terminated environments
        overlay[:, :, 0] = np.minimum(overlay[:, :, 0] + 50, 255)
        return overlay
    
    def _add_title_and_progress_bar(self, grid_image):
        """Add title and progress bar to the top of the grid image."""
        if grid_image is None:
            return None
        
        try:
            import pygame
            
            # Constants for the header
            header_height = 80
            progress_bar_height = 20
            progress_bar_margin = 10
            title_height = 30
            
            # Create new image with header space
            grid_height, grid_width, channels = grid_image.shape
            total_height = grid_height + header_height
            final_image = np.zeros((total_height, grid_width, channels), dtype=grid_image.dtype)
            
            # Fill header with white background
            final_image[:header_height, :, :] = 255
            
            # Copy grid image below the header
            final_image[header_height:, :, :] = grid_image
            
            # Create a temporary pygame surface for text rendering
            temp_surface = pygame.Surface((grid_width, header_height))
            temp_surface.fill((255, 255, 255))  # White background
            
            # Initialize font if needed
            pygame.font.init()
            font = pygame.font.Font(None, 36)
            
            # Draw title
            title_text = font.render("2-Population Lotka-Volterra Environment", True, (0, 0, 0))
            title_rect = title_text.get_rect(center=(grid_width // 2, title_height // 2))
            temp_surface.blit(title_text, title_rect)
            
            # Draw progress bar
            progress_bar_width = grid_width - 2 * progress_bar_margin
            progress_bar_x = progress_bar_margin
            progress_bar_y = title_height + 10
            
            # Background of progress bar (light gray)
            pygame.draw.rect(temp_surface, (200, 200, 200), 
                           (progress_bar_x, progress_bar_y, progress_bar_width, progress_bar_height))
            
            # Progress fill (green to red gradient based on progress)
            if hasattr(self, 'pop_norm'):
                progress = min(self.pop_norm / 1.2, 1.0)  # Normalize to 0-1 range
                fill_width = int(progress_bar_width * progress)
                
                # Color transition from green to red as progress increases
                if progress < 0.5:
                    # Green to yellow
                    red = int(255 * (progress * 2))
                    green = 255
                    blue = 0
                else:
                    # Yellow to red
                    red = 255
                    green = int(255 * (2 - progress * 2))
                    blue = 0
                
                if fill_width > 0:
                    pygame.draw.rect(temp_surface, (red, green, blue),
                                   (progress_bar_x, progress_bar_y, fill_width, progress_bar_height))
            
            # Progress bar border
            pygame.draw.rect(temp_surface, (0, 0, 0),
                           (progress_bar_x, progress_bar_y, progress_bar_width, progress_bar_height), 2)
            
            # Add progress text
            if hasattr(self, 'pop_norm'):
                progress_text = f"Population Norm: {self.pop_norm:.2f} / 1.20"
                progress_font = pygame.font.Font(None, 24)
                progress_surface = progress_font.render(progress_text, True, (0, 0, 0))
                progress_rect = progress_surface.get_rect(center=(grid_width // 2, progress_bar_y + progress_bar_height + 15))
                temp_surface.blit(progress_surface, progress_rect)
            
            # Convert pygame surface to numpy array and update final image
            header_array = pygame.surfarray.array3d(temp_surface).swapaxes(0, 1)
            final_image[:header_height, :, :] = header_array
            
            return final_image
            
        except ImportError:
            logging.error("pygame not available for rendering title and progress bar")
            return grid_image
        except Exception as e:
            logging.error(f"Error adding title and progress bar: {e}")
            return grid_image
    
    def _render_human(self, grid_image):
        """Render the grid image to the pygame display."""
        try:
            import pygame
            
            if self.screen is None:
                return None
            
            # Convert numpy array to pygame surface
            surf = pygame.surfarray.make_surface(grid_image.swapaxes(0, 1))
            self.screen.blit(surf, (0, 0))
            
            # Handle pygame events
            pygame.event.pump()
            if self.clock:
                self.clock.tick(self.metadata['render_fps'])
            pygame.display.flip()
            
            return None  # Human mode doesn't return anything
            
        except ImportError:
            logging.error("pygame not available for human rendering")
            return grid_image
    
    def _construct_obs_from_state(self, env):
        """Manually construct observation from environment state."""
        # For your custom ContinuousCartPoleEnv, the observation is just the state
        if hasattr(env.unwrapped, 'state') and env.unwrapped.state is not None:
            return np.array(env.unwrapped.state, dtype=np.float64)
        
        # Fallback for other environments
        env_type = type(env.unwrapped).__name__
        
        if 'CartPole' in env_type:
            return np.array(env.unwrapped.state, dtype=np.float64)
        elif 'Pendulum' in env_type:
            return np.array(env.unwrapped.state, dtype=np.float64)
        elif 'MountainCar' in env_type:
            return np.array(env.unwrapped.state, dtype=np.float64)
        elif 'Acrobot' in env_type:
            return np.array(env.unwrapped.state, dtype=np.float64)
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
            
        # Add other environment types as needed
        
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

    def close(self):
        """Close all environments and clean up rendering resources."""
        for env in self.envs:
            env.close()
        
        # Clean up pygame resources
        if self.is_rendering_setup:
            try:
                import pygame
                if pygame.get_init():
                    pygame.display.quit()
                    pygame.quit()
            except ImportError:
                pass
        
        self.screen = None
        self.clock = None
        self.is_rendering_setup = False