from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import torch
from typing import Dict, Any

class CustomTensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to TensorBoard during RL training.
    Based on best practices from RL experimentation guidelines.
    """
    
    def __init__(self, 
                 log_frequency: int = 1000,
                 track_gradients: bool = False,
                 track_data_stats: bool = True,
                 verbose: int = 0):
        """
        Args:
            log_frequency: How often to log custom metrics (in steps)
            track_gradients: Whether to track gradient norms (can be expensive)
            track_data_stats: Whether to track observation/reward statistics
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.log_frequency = log_frequency
        self.track_gradients = track_gradients
        self.track_data_stats = track_data_stats
        
        # Buffers for tracking statistics
        self.recent_observations = []
        self.recent_rewards = []
        
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout (data collection phase)"""
        
        # === Episode Statistics ===
        if len(self.model.ep_info_buffer) > 0:
            # Debug: Print what we're getting
            if self.verbose >= 2:
                print(f"Episode buffer size: {len(self.model.ep_info_buffer)}")
                print(f"Sample episode info: {self.model.ep_info_buffer[-1] if self.model.ep_info_buffer else 'None'}")
            
            # Extract episode rewards and lengths
            ep_rewards = []
            ep_lengths = []
            
            for ep_info in self.model.ep_info_buffer:
                if isinstance(ep_info, dict):
                    # Use 'r' for reward, 'l' for length (standard SB3 format)
                    if 'r' in ep_info and 'l' in ep_info:
                        ep_rewards.append(ep_info['r'])
                        ep_lengths.append(ep_info['l'])
                    # Sometimes the keys might be different
                    elif 'episode' in ep_info and isinstance(ep_info['episode'], dict):
                        if 'r' in ep_info['episode']:
                            ep_rewards.append(ep_info['episode']['r'])
                        if 'l' in ep_info['episode']:
                            ep_lengths.append(ep_info['episode']['l'])
            
            if ep_rewards and ep_lengths:
                # Enhanced episode statistics
                self.logger.record('rollout/ep_rew_max', np.max(ep_rewards))
                self.logger.record('rollout/ep_rew_min', np.min(ep_rewards))
                self.logger.record('rollout/ep_rew_std', np.std(ep_rewards))
                self.logger.record('rollout/ep_len_max', np.max(ep_lengths))
                self.logger.record('rollout/ep_len_min', np.min(ep_lengths))
                self.logger.record('rollout/ep_len_std', np.std(ep_lengths))
                
                # Recent performance (last 10 episodes)
                recent_rewards = ep_rewards[-10:] if len(ep_rewards) >= 10 else ep_rewards
                recent_lengths = ep_lengths[-10:] if len(ep_lengths) >= 10 else ep_lengths
                self.logger.record('rollout/recent_ep_rew_mean', np.mean(recent_rewards))
                self.logger.record('rollout/recent_ep_len_mean', np.mean(recent_lengths))
                
                # Debug logging
                if self.verbose >= 1:
                    print(f"Logged episode rewards - Mean: {np.mean(ep_rewards):.2f}, Min: {np.min(ep_rewards):.2f}, Max: {np.max(ep_rewards):.2f}")
            else:
                if self.verbose >= 1:
                    print("Warning: No valid episode rewards found in ep_info_buffer")
        
        # === Algorithm-Specific Metrics (PPO) ===
        try:
            self._log_ppo_metrics()
        except Exception as e:
            if self.verbose >= 1:
                print(f"Warning: Could not log PPO metrics: {e}")
        
        # === Data Quality Metrics ===
        if self.track_data_stats:
            self._log_data_statistics()
            
        # === Gradient Information ===
        if self.track_gradients:
            self._log_gradient_stats()
    
    def _log_ppo_metrics(self):
        """Log PPO-specific metrics"""
        # Policy entropy - crucial for monitoring exploration
        try:
            # Sample some observations to compute entropy
            if hasattr(self.training_env, 'get_attr') and hasattr(self.training_env, 'reset'):
                # For vectorized environments
                obs = self.training_env.reset()
            else:
                obs = self.model.env.reset()
                
            if isinstance(obs, tuple):
                obs = obs[0]  # Handle environments that return (obs, info)
                
            # Convert to tensor if needed
            if not isinstance(obs, torch.Tensor):
                obs = torch.as_tensor(obs, device=self.model.device)
                
            # Ensure correct shape for vectorized env
            if len(obs.shape) == 1:
                obs = obs.unsqueeze(0)
            
            # Get policy distribution
            with torch.no_grad():
                features = self.model.policy.extract_features(obs)
                latent_pi = self.model.policy.mlp_extractor.forward_actor(features)
                distribution = self.model.policy.action_dist.proba_distribution(
                    self.model.policy.action_net(latent_pi)
                )
                entropy = distribution.entropy().mean().item()
                self.logger.record('train/policy_entropy', entropy)
                
        except Exception as e:
            if self.verbose >= 1:
                print(f"Could not compute policy entropy: {e}")
        
        # Access recent training metrics if available
        if hasattr(self.model, '_last_dones') and hasattr(self.model, 'rollout_buffer'):
            # Log rollout buffer statistics
            if self.model.rollout_buffer.full:
                returns = self.model.rollout_buffer.returns.flatten()
                values = self.model.rollout_buffer.values.flatten()
                
                # Explained variance - key diagnostic for value function quality
                var_y = np.var(returns)
                explained_var = 1 - np.var(returns - values) / (var_y + 1e-8)
                self.logger.record('train/explained_variance', explained_var)
                
                # Value function statistics
                self.logger.record('train/value_mean', np.mean(values))
                self.logger.record('train/value_std', np.std(values))
                self.logger.record('train/return_mean', np.mean(returns))
                self.logger.record('train/return_std', np.std(returns))
    
    def _log_data_statistics(self):
        """Log observation and reward statistics"""
        try:
            if hasattr(self.model, 'rollout_buffer') and self.model.rollout_buffer.full:
                # Observation statistics
                obs = self.model.rollout_buffer.observations
                if obs is not None:
                    obs_flat = obs.reshape(-1, obs.shape[-1]) if len(obs.shape) > 2 else obs
                    
                    self.logger.record('data/obs_mean', np.mean(obs_flat))
                    self.logger.record('data/obs_std', np.std(obs_flat))
                    self.logger.record('data/obs_min', np.min(obs_flat))
                    self.logger.record('data/obs_max', np.max(obs_flat))
                
                # Reward statistics
                rewards = self.model.rollout_buffer.rewards
                if rewards is not None:
                    rewards_flat = rewards.flatten()
                    self.logger.record('data/reward_mean', np.mean(rewards_flat))
                    self.logger.record('data/reward_std', np.std(rewards_flat))
                    self.logger.record('data/reward_min', np.min(rewards_flat))
                    self.logger.record('data/reward_max', np.max(rewards_flat))
                    
        except Exception as e:
            if self.verbose >= 1:
                print(f"Could not log data statistics: {e}")
    
    def _log_gradient_stats(self):
        """Log gradient norms for debugging optimization"""
        try:
            total_norm = 0.0
            param_count = 0
            
            for param in self.model.policy.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            
            if param_count > 0:
                total_norm = total_norm ** (1. / 2)
                self.logger.record('train/gradient_norm', total_norm)
                
        except Exception as e:
            if self.verbose >= 1:
                print(f"Could not log gradient statistics: {e}")
    
    def _on_step(self) -> bool:
        """Called after each environment step"""
        # Log less frequent metrics to avoid overwhelming TensorBoard
        if self.n_calls % self.log_frequency == 0:
            # Learning rate tracking
            if hasattr(self.model, 'lr_schedule'):
                current_lr = self.model.lr_schedule(self.model._current_progress_remaining)
                self.logger.record('train/learning_rate', current_lr)
            
            # Environment step statistics
            if hasattr(self.locals, 'infos') and self.locals['infos']:
                # Log any custom info from environment
                for info in self.locals['infos']:
                    if isinstance(info, dict):
                        for key, value in info.items():
                            if isinstance(value, (int, float)) and key not in ['episode', 'terminal_observation']:
                                self.logger.record(f'env/{key}', value)
        
        return True
    
    def _on_training_start(self) -> None:
        """Called at the beginning of training"""
        if self.verbose >= 1:
            print("CustomTensorboardCallback: Starting enhanced logging")
    
    def _on_training_end(self) -> None:
        """Called at the end of training"""
        if self.verbose >= 1:
            print("CustomTensorboardCallback: Training completed")