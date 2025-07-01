import gymnasium as gym
import numpy as np
from stable_baselines3.common.running_mean_std import RunningMeanStd

class NormalizationWrapper(gym.Wrapper):
    def __init__(self, env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0):
        super().__init__(env)
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        
        if norm_obs:
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
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
        self.obs_rms.update(obs)
        normalized = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        return np.clip(normalized, -self.clip_obs, self.clip_obs)
        
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