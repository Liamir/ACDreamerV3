from typing import Optional
import numpy as np
import gymnasium as gym
import copy
import math

class ProstateCancerTherapyEnv(gym.Env):
    """
    Ordering of cell types is always: T+, TP, T- (which follows the original paper)
    https://www.nature.com/articles/s41467-017-01968-5
    """

    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30,
    }

    def __init__(self, LV_params, dt=0.01, render_mode='rgb_array', k=1):
        self.render_mode = render_mode
        self.dt = dt
        self.k = k
        self.params = copy.deepcopy(LV_params)
        self.tp_cap_on_treatment = self.params['tp_cap_on_treatment']
        self.tp_capacity_off_treatment = self.params['carrying_capacities'][1]
        self.growth_rates = self.params['growth_rates']
        self.competition_matrix = self.params['competition_matrix']
        
        # T+ counts, TP counts, T- counts
        self.counts_low = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.counts_high = np.array([10000.0, 10000.0, 10000.0], dtype=np.float64)
        self.population_low = np.array([0.0], dtype=np.float64)
        self.population_high = np.array([10000.0], dtype=np.float64)

        self.observation_space = gym.spaces.Dict(
            {
                "counts": gym.spaces.Box(self.counts_low, self.counts_high, dtype=np.float64),
                "population": gym.spaces.Box(self.population_low, self.population_high, dtype=np.float64),
            }
        )

        # 0 is off treatment, 1 is on treatment
        self.action_space = gym.spaces.Discrete(2)
    
    def _get_obs(self):
        return {
            "counts": self.counts,
            "population": np.array([self.population_size]),
        }

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
        """Start a new episode

        """
        super().reset(seed=seed)

        self.init_options = options or {}

        should_randomize = (
            'low' in self.init_options and 'high' in self.init_options and
            any(key in self.init_options['low'] for key in ['tplus_counts', 'tprod_counts', 'tneg_counts'])
        )
        
        if should_randomize:
            low_ranges = self.init_options['low']
            high_ranges = self.init_options['high']
            
            tplus_min = low_ranges['tplus_counts']
            tplus_max = high_ranges['tplus_counts']
            tplus_count = self.np_random.uniform(tplus_min, tplus_max)
            
            tprod_min = low_ranges['tprod_counts']
            tprod_max = high_ranges['tprod_counts']
            tprod_count = self.np_random.uniform(tprod_min, tprod_max)
            
            tneg_min = low_ranges['tneg_counts']
            tneg_max = high_ranges['tneg_counts']
            tneg_count = self.np_random.uniform(tneg_min, tneg_max)
            
            # Set the randomized counts
            self.counts = np.array([tplus_count, tprod_count, tneg_count])
            self.population_size = np.sum(self.counts)
            
            # print(f"Randomized initialization:")
            # print(f"  T+ cells: {tplus_count:.1f}")
            # print(f"  TP cells: {tprod_count:.1f}")
            # print(f"  T- cells: {tneg_count:.1f}")
            # print(f"  Total population: {self.population_size:.1f}")
            
        elif any(key in self.init_options for key in ['tplus_counts', 'tprod_counts', 'tneg_counts']):
            tplus_count = self.init_options['tplus_counts']
            tprod_count = self.init_options['tprod_counts']
            tneg_count = self.init_options['tneg_counts']
            self.counts = np.array([tplus_count, tprod_count, tneg_count])
            self.population_size = np.sum(self.counts)
        
        else:  # Use default initialization from fixed LV_params
            self.population_size = np.sum(self.params['init_counts'])
            self.counts = self.params['init_counts']

        self.original_population = self.population_size

        self.carrying_capacities = self.params['carrying_capacities']
        self.carrying_capacities[0] = 1.5 * self.counts[1]

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        """
        Execute one timestep within the environment
        """

        self.last_action = action

        # assume for now there is no treatment
        if action == 0:  # off treatment
            self.carrying_capacities[0] = 1.5 * self.counts[1]
            self.carrying_capacities[1] = self.tp_capacity_off_treatment
        elif action == 1:  # on treatment
            self.carrying_capacities[0] = 0.5 * self.counts[1]
            self.carrying_capacities[1] = self.tp_cap_on_treatment
        else:
            raise ValueError(f'Illegal action: {action}')
        # apply the LV equation
        competition_sums = np.dot(self.competition_matrix, self.counts)  # Shape: (3,)
        competition_effects = competition_sums / self.carrying_capacities  # Shape: (3,)
        self.counts += self.dt * self.growth_rates * self.counts * (1 - competition_effects)
        self.counts = np.where(self.counts < 1.0e-9, 1.0e-9, self.counts)
        self.population_size = self.counts.sum()

        truncated = False
        terminated = False

        reward = 0.0

        # reward given for every step before progression
        reward += 0.0005

        # small penalty given for treatment
        # if action == 1:
        #     reward -= 0.0001

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
    
    def render(self):
        """
        Render the prostate cancer therapy environment using pygame.
        Shows PSA level as a vertical bar and cell distribution as a pie chart.
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

        # Screen dimensions - very compact layout
        screen_width = 250
        screen_height = 250
        
        if not hasattr(self, 'screen') or self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
                pygame.display.set_caption("Prostate Cancer Therapy Environment")
            else:  # rgb_array mode
                self.screen = pygame.Surface((screen_width, screen_height))
        
        if not hasattr(self, 'clock') or self.clock is None:
            self.clock = pygame.time.Clock()

        # Create the main surface
        surf = pygame.Surface((screen_width, screen_height))
        surf.fill((255, 255, 255))  # White background

        # Font for text rendering
        if not hasattr(self, 'font') or self.font is None:
            self.font = pygame.font.Font(None, 26)
            self.small_font = pygame.font.Font(None, 22)

        # === Cell Distribution Pie Chart (Centered with minimal margins) ===
        pie_center_x = screen_width // 2
        pie_center_y = screen_height // 2 - 5  # Slightly higher to balance with text below
        pie_radius = 70  # Slightly larger pie chart
        
        # Calculate ratios
        ratios = self.counts / self.population_size
        
        # Colors: T+ (light blue), TP (medium blue), T- (red)
        colors = [(123, 179, 240), (74, 144, 226), (231, 76, 60)]
        labels = ["T+", "TP", "T-"]
        
        # === Treatment Status Indicator (above pie chart, minimal margin) ===
        treatment_status = "Treatment: Unknown"
        if hasattr(self, 'last_action'):
            treatment_status = f"Treatment: {'ON' if self.last_action == 1 else 'OFF'}"
        
        status_color = (0, 150, 0) if hasattr(self, 'last_action') and self.last_action == 1 else (150, 0, 0)
        status_text = self.font.render(treatment_status, True, status_color)
        status_rect = status_text.get_rect(center=(pie_center_x, pie_center_y - pie_radius - 15))
        surf.blit(status_text, status_rect)

        # Draw pie chart
        start_angle = -90  # Start from top (12 o'clock position)
        for i, (ratio, color, label) in enumerate(zip(ratios, colors, labels)):
            if ratio > 0:
                end_angle = start_angle + ratio * 360
                
                # Draw pie slice
                points = [pie_center_x, pie_center_y]
                for angle in range(int(start_angle), int(end_angle) + 1):
                    x = pie_center_x + pie_radius * math.cos(math.radians(angle))
                    y = pie_center_y + pie_radius * math.sin(math.radians(angle))
                    points.extend([x, y])
                
                if len(points) >= 6:  # Need at least 3 points for a polygon
                    points_tuples = [(points[i], points[i+1]) for i in range(0, len(points), 2)]
                    gfxdraw.filled_polygon(surf, points_tuples, color)
                    gfxdraw.aapolygon(surf, points_tuples, (0, 0, 0))
                
                # Add percentage text
                if ratio > 0.06:  # Show text if slice is big enough
                    mid_angle = (start_angle + end_angle) / 2
                    text_x = pie_center_x + (pie_radius * 0.65) * math.cos(math.radians(mid_angle))
                    text_y = pie_center_y + (pie_radius * 0.65) * math.sin(math.radians(mid_angle))
                    
                    percentage_text = self.font.render(f"{ratio*100:.1f}%", True, (255, 255, 255))
                    text_rect = percentage_text.get_rect(center=(text_x, text_y))
                    surf.blit(percentage_text, text_rect)
                
                start_angle = end_angle
        
        # Draw pie chart border
        gfxdraw.aacircle(surf, pie_center_x, pie_center_y, pie_radius, (0, 0, 0))
        
        # Legend positioned below pie chart, with more breathing room
        legend_start_x = pie_center_x - 60  # Tight centering
        legend_start_y = pie_center_y + pie_radius + 15
        
        for i, (color, label) in enumerate(zip(colors, labels)):
            legend_x = legend_start_x + i * 40  # Very close spacing
            
            # Smaller color box
            pygame.draw.rect(surf, color, (legend_x, legend_start_y, 12, 12))
            pygame.draw.rect(surf, (0, 0, 0), (legend_x, legend_start_y, 12, 12), 1)
            
            # Shortened label names
            label_names = ["T+", "TP", "T-"]
            label_text = self.small_font.render(label_names[i], True, (0, 0, 0))
            surf.blit(label_text, (legend_x + 15, legend_start_y - 2))
            
            # Cell count for this type (smaller font)
            count_text = self.small_font.render(f"{self.counts[i]:.0f}", True, (100, 100, 100))
            surf.blit(count_text, (legend_x + 15, legend_start_y + 12))

        # Total cell count below legend, with more space
        count_y = legend_start_y + 35
        count_text = self.small_font.render(f"Total: {self.population_size:.0f}", True, (0, 0, 0))
        count_rect = count_text.get_rect(center=(pie_center_x, count_y))
        surf.blit(count_text, count_rect)

        # Blit everything to the main screen
        self.screen.blit(surf, (0, 0))
        
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(30)  # 30 FPS
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