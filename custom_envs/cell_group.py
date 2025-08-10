from typing import Optional
import numpy as np
import gymnasium as gym
import copy

class ProstateCancerTherapyEnv(gym.Env):
    """
    Ordering of cell types is always: T+, TP, T- (which follows the original paper)
    https://www.nature.com/articles/s41467-017-01968-5
    """
    def __init__(self, LV_params, dt=0.01, init_pop_ratio=0.4):
        self.dt = dt
        self.init_pop_ratio = init_pop_ratio
        self.params = copy.deepcopy(LV_params)
        self.tp_cap_on_treatment = self.params['tp_cap_on_treatment']
        self.growth_rates = self.params['growth_rates']
        self.competition_matrix = self.params['competition_matrix']
        self.ess_psa = self.params['ess_psa']

        # initialized in reset()
        self.carrying_capacities = [-1, -1, -1]
        self.counts = None
        self.population_size = -1
        self.psa_norm = -1
        self.psa = -1
        
        # the observation should be the counts of all cell types.
        # these can be either direct count data, or ratio with another total population entry.
        # T+ ratio, TP ratio, T- ratio, total population
        self.ratios_low = np.array([0, 0, 0], dtype=np.float64)
        self.ratios_high = np.array([1, 1, 1], dtype=np.float64)
        self.population_low = np.array([0], dtype=np.float64)
        self.population_high = np.array([np.inf], dtype=np.float64)
        self.psa_low = np.array([0], dtype=np.float64)
        self.psa_high = np.array([np.inf], dtype=np.float64)

        # 0 is off treatment, 1 is on treatment
        self.action_space = gym.spaces.Discrete(2)

        self.observation_space = gym.spaces.Dict(
            {
                "ratios": gym.spaces.Box(self.ratios_low, self.ratios_high, dtype=np.float64),
                "population": gym.spaces.Box(self.population_low, self.population_high, dtype=np.float64),
                "psa": gym.spaces.Box(self.psa_low, self.psa_high, dtype=np.float64),
            }
            
        )
    
    def _get_obs(self):
        return {
            "ratios": self.counts / self.population_size,
            "population": np.array([self.population_size]),
            "psa": self.psa_norm,
        }

    def _get_counts(self):
        """
        Returns: Cell type counts
        """
        return self.population_size * self._get_obs()["ratios"]


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

        self.population_size = np.sum(self.init_pop_ratio * self.params['ess_counts'])
        self.counts = self.init_pop_ratio * self.params['ess_counts']

        self.psa = self.init_pop_ratio * float(self.ess_psa)
        print('in reset', self.psa, type(self.psa))
        self.original_psa = self.psa
        self.psa_norm = np.float64(1.0)

        self.carrying_capacities = self.params['carrying_capacities']
        self.carrying_capacities[0] = 1.5 * self.counts[1]
        self.tp_capacity_off_treatment = self.carrying_capacities[1]

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        """Execute one timestep within the environment
        """
        prev_counts = self.counts.copy()

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
        # counts = self._get_counts()
        competition_sums = np.dot(self.competition_matrix, self.counts)  # Shape: (3,)
        competition_effects = competition_sums / self.carrying_capacities  # Shape: (3,)
        self.counts += self.dt * self.growth_rates * self.counts * (1 - competition_effects)
        self.counts = np.where(self.counts < 1.0e-9, 1.0e-9, self.counts)
        self.population_size = self.counts.sum()

        # update PSA
        s = np.sum(prev_counts)
        # print(self.dt, type(self.dt))
        # print(s, type(s))
        # print('before', self.psa, type(self.psa))
        self.psa += self.dt * (np.sum(prev_counts) - 0.5 * self.psa)
        # print('after', self.psa, type(self.psa))
        self.psa_norm = self.psa / self.original_psa

        terminated = False
        truncated = False

        # a constant size tumor gets 0 reward,
        # shrinked tumor gets a positive reward, and an increase in tumor size gets negative reward
        # calculating reward based on the new state (new PSA)
        reward = (1 - self.psa_norm)  # multiply by some factor?

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info