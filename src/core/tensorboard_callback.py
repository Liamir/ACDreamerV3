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
                 track_step_rewards: bool = False,  # New parameter
                 verbose: int = 0):
        """
        Args:
            log_frequency: How often to log custom metrics (in steps)
            track_gradients: Whether to track gradient norms (can be expensive)
            track_data_stats: Whether to track observation/reward statistics
            track_step_rewards: Whether to track step-wise reward stats (disable for constant reward envs)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.log_frequency = log_frequency
        self.track_gradients = track_gradients
        self.track_data_stats = track_data_stats
        self.track_step_rewards = track_step_rewards
        
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
            # Debug: Check what attributes the model actually has
            if self.verbose >= 2:
                print(f"Model type: {type(self.model)}")
                print(f"Model attributes: {[attr for attr in dir(self.model) if 'buffer' in attr.lower()]}")
                print(f"Has rollout_buffer: {hasattr(self.model, 'rollout_buffer')}")
                print(f"Has _last_dones: {hasattr(self.model, '_last_dones')}")
            
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
            
            # Get policy distribution - FIXED VERSION
            with torch.no_grad():
                # Use the proper SB3 method that handles distribution creation correctly
                distribution = self.model.policy.get_distribution(obs)
                entropy = distribution.entropy().mean().item()
                self.logger.record('train/policy_entropy', entropy)
                
                if self.verbose >= 2:
                    print(f"Logged policy entropy: {entropy:.4f}")
                    
        except Exception as e:
            if self.verbose >= 1:
                print(f"Could not compute policy entropy: {e}")
            
            # Fallback method: Try to get entropy from rollout buffer if available
            try:
                if hasattr(self.model, 'rollout_buffer') and self.model.rollout_buffer.full:
                    # This is less accurate but still useful
                    actions = self.model.rollout_buffer.actions
                    if actions is not None:
                        # For continuous actions, estimate entropy from action variance
                        action_std = torch.std(actions.flatten()).item()
                        # Rough entropy estimate for Gaussian: 0.5 * log(2 * pi * e * var)
                        estimated_entropy = 0.5 * np.log(2 * np.pi * np.e * action_std**2)
                        self.logger.record('train/policy_entropy_estimated', estimated_entropy)
                        
                        if self.verbose >= 2:
                            print(f"Logged estimated entropy: {estimated_entropy:.4f}")
            except Exception as e2:
                if self.verbose >= 1:
                    print(f"Fallback entropy computation also failed: {e2}")
        
        # Access recent training metrics if available
        # Try different ways to access the rollout buffer
        rollout_buffer = None
        
        # Method 1: Direct access
        if hasattr(self.model, 'rollout_buffer'):
            rollout_buffer = self.model.rollout_buffer
            if self.verbose >= 2:
                print("Found rollout_buffer via direct access")
        
        # Method 2: Check if it's in the policy
        elif hasattr(self.model, 'policy') and hasattr(self.model.policy, 'rollout_buffer'):
            rollout_buffer = self.model.policy.rollout_buffer
            if self.verbose >= 2:
                print("Found rollout_buffer via policy")
        
        # Method 3: Check common SB3 locations
        elif hasattr(self.model, '_last_obs') and hasattr(self.model, 'num_timesteps'):
            # Model exists but no rollout_buffer - might be timing issue
            if self.verbose >= 2:
                print("Model exists but no rollout_buffer found")
        
        if rollout_buffer is not None:
            if self.verbose >= 2:
                print(f"Rollout buffer full: {rollout_buffer.full}")
                print(f"Buffer size: {rollout_buffer.buffer_size}")
                print(f"Buffer pos: {rollout_buffer.pos}")
                
            if rollout_buffer.full:
                try:
                    returns = rollout_buffer.returns.flatten()
                    values = rollout_buffer.values.flatten()
                    
                    if self.verbose >= 2:
                        print(f"Returns shape: {returns.shape}, Values shape: {values.shape}")
                    
                    # Value function statistics
                    self.logger.record('custom/value_mean', np.mean(values))
                    self.logger.record('custom/value_std', np.std(values))
                    self.logger.record('custom/return_mean', np.mean(returns))
                    self.logger.record('custom/return_std', np.std(returns))
                    
                    if self.verbose >= 1:
                        print(f"Logged value metrics")
                        
                except Exception as e:
                    if self.verbose >= 1:
                        print(f"Error accessing rollout buffer data: {e}")
            else:
                if self.verbose >= 2:
                    print("Rollout buffer not full yet")
        else:
            if self.verbose >= 1:
                print("No rollout buffer found - value metrics not available")


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
                
                # Reward statistics (only if enabled and rewards vary)
                if self.track_step_rewards:
                    rewards = self.model.rollout_buffer.rewards
                    if rewards is not None:
                        rewards_flat = rewards.flatten()
                        # Only log if rewards have meaningful variation
                        if np.std(rewards_flat) > 1e-6:
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