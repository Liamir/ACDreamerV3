import gymnasium as gym
import numpy as np
from stable_baselines3.common.running_mean_std import RunningMeanStd

class NormalizationWrapper(gym.Wrapper):
    def __init__(self, env, norm_obs=True, norm_reward=True, clip_reward=10.0, cfg=None):
        super().__init__(env)
        self.agent_type = cfg.agent_type
        self.objective_type = cfg.objective_type
        self.effective_k = env.original_k if self.agent_type == 'spatial' else 1
        self.single_obs_size = 2
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.clip_reward = clip_reward

        if norm_reward:
            self.ret_rms = RunningMeanStd(shape=())

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped environment"""
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return getattr(self.env, name)
            
    def _normalize_obs(self, obs):
        if not self.norm_obs:
            return obs
        
        # counts normalization
        # obs size (np array): 4K + 1: 4 * (c11, c12, c13, p) + total_p
        for i in range(0, self.single_obs_size * self.effective_k, self.single_obs_size):
            obs[i : i + self.single_obs_size] = ((np.log10(obs[i : i + self.single_obs_size]) - (-9.0)) / (3.0 - (-9.0))) * 2 - 1

        # total population normalization
        original_population = self.env.total_initial_population
        if self.objective_type == 'TTP':
            # total p should be in the range (1e2, 1e5)
            # div by original total - must be in the range (0, 1.2), because of termination
            obs[-1] = (obs[-1] / original_population) * (2.0 / 1.2) - 1
        
        return obs
        
    def _normalize_reward(self, reward):
        if not self.norm_reward:
            return reward
        self.ret_rms.update(np.array([reward]))
        normalized = reward / np.sqrt(self.ret_rms.var + 1e-8)
        return np.clip(normalized, -self.clip_reward, self.clip_reward)
            
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.norm_obs:
            obs = self._normalize_obs(obs)
        return obs, info
        
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        if self.norm_obs:
            obs = self._normalize_obs(obs)
        if self.norm_reward:
            reward = self._normalize_reward(reward)
        return obs, reward, done, truncated, info