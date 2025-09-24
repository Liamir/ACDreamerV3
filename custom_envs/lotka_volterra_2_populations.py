from typing import Optional
import numpy as np
import gymnasium as gym
import copy
import math

class LV2PopulationsEnv(gym.Env):

    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30,
    }

    def __init__(self, LV_params, cfg, render_mode='rgb_array'):
        self.render_mode = render_mode
        self.params = copy.deepcopy(LV_params)
        self.cfg = cfg

        # ODE parameters
        self.dt = LV_params['dt']
        self.growth_rates = np.array(LV_params['growth_rates'])
        self.death_rates = np.array(LV_params['death_rates'])
        self.carrying_capacity = LV_params['carrying_capacity']
        self.drug_efficiency = LV_params['drug_efficiency']
        
        # Sensitive (S) counts and Resistant (R) counts
        counts_low = np.array([0.0] * 2, dtype=np.float64)
        counts_high = np.array([self.carrying_capacity * 2] * 2, dtype=np.float64)
        self.observation_space = gym.spaces.Box(counts_low, counts_high, dtype=np.float64)
        
        # 0 is off treatment, 1 is on treatment
        self.action_space = gym.spaces.Discrete(2)
    
    def _get_obs(self):
        # if self.cfg.observation_type == 'stacked':
        #     return self.counts
        # elif self.cfg.observation_type == 'stacked_pop_norm':
        #     return np.concatenate([self.counts, np.array([self.population_size])])
        return self.counts

    def _get_counts(self):
        """
        Returns: Cell type counts
        """
        return self.counts

    def _get_info(self):
        """
        Returns: Cell type counts
        """
        return {
            "counts": self._get_counts(),
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[int] = None):

        super().reset(seed=seed)

        self.init_options = options or {}

        should_randomize = (
            'low' in self.init_options and 'high' in self.init_options and
            any(key in self.init_options['low'] for key in ['s_counts', 'r_counts'])
        )
        
        if should_randomize:
            low_ranges = self.init_options['low']
            high_ranges = self.init_options['high']
            
            s_min = low_ranges['s_counts']
            s_max = high_ranges['s_counts']
            s_count = self.np_random.uniform(s_min, s_max)
            
            r_min = low_ranges['r_counts']
            r_max = high_ranges['r_counts']
            r_count = self.np_random.uniform(r_min, r_max)
            
            # Set the randomized counts
            self.counts = np.array([s_count, r_count])
            self.population_size = np.sum(self.counts)
            
            # print(f"Randomized initialization:")
            # print(f"  T+ cells: {tplus_count:.1f}")
            # print(f"  TP cells: {tprod_count:.1f}")
            # print(f"  T- cells: {tneg_count:.1f}")
            # print(f"  Total population: {self.population_size:.1f}")
            
        elif any(key in self.init_options for key in ['s_counts', 'r_counts']):
            s_count = self.init_options['s_counts']
            r_count = self.init_options['r_counts']
            self.counts = np.array([s_count, r_count])
            self.population_size = np.sum(self.counts)
        
        else:  # Use default initialization from fixed LV_params
            self.population_size = np.sum(self.params['init_counts'])
            self.counts = self.params['init_counts']

        self.initial_population_size = self.population_size

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        """
        Execute one timestep within the environment
        """

        self.last_action = action

        # apply the LV equation
        dS = self.growth_rates[0] * self.counts[0] * (1 - (np.sum(self.counts) / self.carrying_capacity)) * (1 - self.drug_efficiency * action) - self.death_rates[0] * self.counts[0]
        dR = self.growth_rates[1] * self.counts[1] * (1 - (np.sum(self.counts) / self.carrying_capacity)) - self.death_rates[1] * self.counts[1]
        self.counts += np.array([dS, dR], dtype=np.float64) * self.dt
        self.counts = np.where(self.counts < 1.0e-9, 1.0e-9, self.counts)
        self.population_size = self.counts.sum()

        truncated = False
        terminated = False

        reward = 0.0

        if self.cfg.reward_type == 'TTP':
            # reward given for every step before progression
            # TODO: tune the reward according to the MTD trajectory
            reward += 0.0005

        elif self.cfg.reward_type == 'TB':
            # if self.population_size / self.initial_population_size < 1.2:  # below progression threshold
                # print(f'adding reward :)')
                # reward += 0.01
            # prev approach: give reward purely based on population size
            reward -= self.population_size / (100 * self.carrying_capacity)

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
    
    
    def render(self):
        """
        Render the 2-population Lotka-Volterra environment using pygame.
        Shows cell distribution as a pie chart and treatment status.
        """
        if self.render_mode is None:
            return None

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise gym.error.DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            ) from e

        # Screen dimensions - compact layout
        screen_width = 550
        screen_height = 350
        
        if not hasattr(self, 'screen') or self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
                pygame.display.set_caption("2-Population Lotka-Volterra Environment")
            else:  # rgb_array mode
                self.screen = pygame.Surface((screen_width, screen_height))
        
        if not hasattr(self, 'clock') or self.clock is None:
            self.clock = pygame.time.Clock()

        # Create the main surface
        surf = pygame.Surface((screen_width, screen_height))
        surf.fill((255, 255, 255))  # White background

        # Font for text rendering
        if not hasattr(self, 'font') or self.font is None:
            self.font = pygame.font.Font(None, 22)  # Larger font for treatment status
            self.small_font = pygame.font.Font(None, 18)  # Larger font for counts

        # === Cell Distribution Pie Chart (Moved up toward title) ===
        pie_center_x = screen_width // 2
        pie_center_y = screen_height // 2 - 20  # Moved up more toward title
        pie_radius = 70  # Pie chart radius
        
        # Calculate ratios
        ratios = self.counts / self.population_size if self.population_size > 0 else np.array([0.5, 0.5])
        
        # Colors: Sensitive (blue), Resistant (red)
        colors = [(74, 144, 226), (231, 76, 60)]
        labels = ["S", "R"]
        
        # === Treatment Status Indicator (above pie chart, closer spacing) ===
        treatment_status = "Treatment: OFF"
        if hasattr(self, 'last_action'):
            treatment_status = f"Treatment: {'ON' if self.last_action == 1 else 'OFF'}"
        
        status_color = (0, 150, 0) if hasattr(self, 'last_action') and self.last_action == 1 else (150, 0, 0)
        status_text = self.font.render(treatment_status, True, status_color)
        status_rect = status_text.get_rect(center=(pie_center_x, pie_center_y - pie_radius - 10))  # Closer to pie
        surf.blit(status_text, status_rect)

        # Draw pie chart using pygame.draw.polygon for better compatibility
        start_angle = 0  # Start from right (3 o'clock position)
        for i, (ratio, color) in enumerate(zip(ratios, colors)):
            if ratio > 0.001:  # Only draw if there's a meaningful slice
                # Create points for the pie slice
                points = [(pie_center_x, pie_center_y)]
                
                # Add points around the arc
                num_points = max(int(ratio * 60), 3)  # More points for larger slices
                for j in range(num_points + 1):
                    angle = start_angle + (j / num_points) * ratio * 360
                    x = pie_center_x + pie_radius * math.cos(math.radians(angle))
                    y = pie_center_y + pie_radius * math.sin(math.radians(angle))
                    points.append((x, y))
                
                # Draw filled polygon for pie slice
                if len(points) >= 3:
                    pygame.draw.polygon(surf, color, points)
                    pygame.draw.polygon(surf, (0, 0, 0), points, 2)  # Black border
                
                # Add percentage text
                if ratio > 0.1:  # Show text if slice is big enough
                    mid_angle = start_angle + ratio * 180
                    text_x = pie_center_x + (pie_radius * 0.6) * math.cos(math.radians(mid_angle))
                    text_y = pie_center_y + (pie_radius * 0.6) * math.sin(math.radians(mid_angle))
                    
                    percentage_text = self.font.render(f"{ratio*100:.1f}%", True, (255, 255, 255))
                    text_rect = percentage_text.get_rect(center=(text_x, text_y))
                    surf.blit(percentage_text, text_rect)
                
                start_angle += ratio * 360
        
        # Draw pie chart outer border circle
        pygame.draw.circle(surf, (0, 0, 0), (pie_center_x, pie_center_y), pie_radius, 2)
        
        # Legend positioned below pie chart
        legend_start_x = pie_center_x - 50  # Center the legend for 2 items
        legend_start_y = pie_center_y + pie_radius + 15
        
        for i, (color, label) in enumerate(zip(colors, labels)):
            legend_x = legend_start_x + i * 100  # Spacing between legend items
            
            # Color box
            pygame.draw.rect(surf, color, (legend_x, legend_start_y, 12, 12))
            pygame.draw.rect(surf, (0, 0, 0), (legend_x, legend_start_y, 12, 12), 1)
            
            # Label names - BLACK TEXT for visibility
            label_names = ["Sensitive", "Resistant"]
            label_text = self.font.render(label_names[i], True, (0, 0, 0))
            surf.blit(label_text, (legend_x + 15, legend_start_y - 2))
            
            # Cell count for this type - BLACK for better visibility
            count_text = self.font.render(f"{self.counts[i]:.0f}", True, (0, 0, 0))
            surf.blit(count_text, (legend_x + 15, legend_start_y + 12))

        # Total cell count below legend - BLACK TEXT with larger font
        count_y = legend_start_y + 35
        count_text = self.font.render(f"Total: {self.population_size:.0f}", True, (0, 0, 0))
        count_rect = count_text.get_rect(center=(pie_center_x, count_y))
        surf.blit(count_text, count_rect)

        # Blit everything to the main screen
        self.screen.blit(surf, (0, 0))
        
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(30)
            pygame.display.flip()
            return None
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        """Close the rendering window"""
        if hasattr(self, 'screen') and self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()