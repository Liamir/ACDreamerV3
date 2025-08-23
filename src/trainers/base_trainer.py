"""
Base Trainer Module - Updated with config-based initialization
"""

import os
import re
import random
import numpy as np
import copy
import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from pathlib import Path
from abc import ABC, abstractmethod
import traceback

# Import custom environments and wrappers
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import custom_envs
from custom_envs.action_coupled_wrapper import ActionCoupledWrapper
from custom_envs.normalization_wrapper import NormalizationWrapper
from ..core.tensorboard_callback import CustomTensorboardCallback
from ..core.experiment import ExperimentManager, print_env_info


class BaseTrainer(ABC):
    """Base class for all RL algorithm trainers"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.experiment_manager = ExperimentManager(cfg.experiment.save_path)

    def _get_environment_options(self):
        """Extract initialization options from config"""
        options = {}

        if hasattr(self.cfg.environment, 'init_low') and hasattr(self.cfg.environment, 'init_high'):
            init_low = getattr(self.cfg.environment, 'init_low', None)
            init_high = getattr(self.cfg.environment, 'init_high', None)
            
            if init_low is not None and init_high is not None:
                options['low'] = dict(init_low) if hasattr(init_low, '_asdict') else init_low
                options['high'] = dict(init_high) if hasattr(init_high, '_asdict') else init_high
                
                print(f"Using custom initialization ranges:")
                print(f"  Low: {options['low']}")
                print(f"  High: {options['high']}")
        
        if hasattr(self.cfg.environment, 'init_state'):
            init_state = self.cfg.environment.init_state
            options['init_state'] = dict(init_state) if hasattr(init_state, '_asdict') else init_state
        
        if hasattr(self.cfg.environment, 'reward_type'):
            reward_type = getattr(self.cfg.environment, 'reward_type')
            options['reward_type'] = reward_type

        if hasattr(self.cfg.environment, 'termination_type'):
            termination_type = getattr(self.cfg.environment, 'termination_type')
            options['termination_type'] = termination_type
        
        return options if options else None

    
    def train(self):
        """Train model with checkpoints and evaluation"""
        print(f"Starting {self.cfg.algorithm.name} training for {self.cfg.training.timesteps} steps")
        
        # Create experiment structure
        experiment_path, experiment_id = self.experiment_manager.create_experiment_folder(self.cfg)
        self.experiment_manager.register_experiment(experiment_path, experiment_id, self.cfg)
        self.experiment_manager.save_experiment_config(self.cfg, experiment_path)
        
        print(f"Experiment ID: {experiment_id}")
        print(f"Experiment path: {experiment_path}")
        
        try:
            # Get initialization options from config (prefer this over init_ranges parameter)
            config_options = self._get_environment_options()
            
            # Create environments with config-based options
            env = self._create_environment(config_options)
            eval_env = self._create_environment(config_options, for_evaluation=True)
            
            print_env_info(env, self.cfg)
            
            # Create model (algorithm-specific)
            model = self._create_model(env, experiment_path)
            
            # Setup callbacks
            callbacks = self._create_callbacks(experiment_path, eval_env)
            
            # Train the model
            print(f"Starting training...")
            print(f"Logs and checkpoints will be saved to: {experiment_path}")
            
            model.learn(
                total_timesteps=self.cfg.training.timesteps,
                callback=callbacks,
                progress_bar=True
            )
            
            # Save final model
            final_model_path = Path(experiment_path) / "models" / "final_model"
            model.save(str(final_model_path))
            print(f"Final model saved to: {final_model_path}")
            
            self.experiment_manager.update_experiment_status(experiment_id, "completed")
            
        # except Exception as e:
        #     print(f"Training failed: {e}")
        #     self.experiment_manager.update_experiment_status(experiment_id, "failed", {"error": str(e)})
        #     raise

        except Exception as e:
            error_message = traceback.format_exc()
            print(f"Training failed with traceback:\n{error_message}")
            self.experiment_manager.update_experiment_status(experiment_id, "failed", {"error": error_message})
            raise
        
        return model

    
    def test(self):
        """Test a trained model"""
        num_episodes = self.cfg.evaluation.episodes
        test_only_fixed_policy = self.cfg.evaluation.test_only_fixed_policy
        
        if test_only_fixed_policy:
            fixed_policy = self.cfg.evaluation.fixed_policy_type
        else:
            # Determine model path
            if self.cfg.evaluation.model_path:
                checkpoint_path = self.cfg.evaluation.model_path
                print(f"Using provided model path: {checkpoint_path}")
            else:
                checkpoint_path = self.experiment_manager.find_model_from_config(self.cfg)
                if not checkpoint_path:
                    print("Could not find model. Please provide model_path or ensure experiment exists.")
                    return
            
            model = self._load_model(checkpoint_path)
            if model is None:
                print("Could not load model.")
                return
        
        config_options = self._get_environment_options()

                
        if not test_only_fixed_policy:
            print(f'Testing model from: {checkpoint_path}')
        print(f'Environment: {self.cfg.experiment.env_import}')
        print(f'Number of episodes: {num_episodes}')
        print("-" * 50)
        
        eval_type = self.cfg.evaluation.eval_type
        if eval_type == "visual":
            env = self._create_environment(config_options, render_mode="human")
            # Test episodes
            if test_only_fixed_policy:
                episode_rewards, episode_steps = self._run_test_episodes(None, env, num_episodes, config_options, fixed_policy=fixed_policy)
            else:
                episode_rewards, episode_steps = self._run_test_episodes(model, env, num_episodes, config_options)

            self._print_test_summary(episode_rewards, episode_steps)

        elif eval_type == "numeric":
            env = self._create_environment(config_options, render_mode="rgb_array")
            episode_rewards, episode_steps, episode_scores = self._compare_test_episodes(model, env, num_episodes, config_options)
        
        env.close()
        return episode_rewards, episode_steps
    
    def resume(self):
        """Resume training from a checkpoint"""
        additional_steps = self.cfg.training.timesteps
        
        # Find the model to resume from
        if self.cfg.evaluation.model_path:
            checkpoint_path = self.cfg.evaluation.model_path
            experiment_path = self._extract_experiment_path_from_model(self.cfg.evaluation.model_path)
        else:
            checkpoint_path = self.experiment_manager.find_model_from_config(self.cfg)
            if not checkpoint_path:
                print("Could not find model to resume from.")
                return None, None
            experiment_path = str(Path(checkpoint_path).parent.parent)
        
        print(f"Resuming training from: {checkpoint_path}")
        print(f"Experiment folder: {experiment_path}")
        
        # Load model
        model = self._load_model(checkpoint_path)
        if model is None:
            return None, None
        
        # Extract current step count
        starting_steps = self._extract_step_count(checkpoint_path, model)
        print(f"Starting from step: {starting_steps}")
        print(f"Will train for {additional_steps} additional steps")
        
        # Get initialization options from config
        config_options = self._get_environment_options()
        
        # Create environments
        env = self._create_environment(config_options)
        eval_env = self._create_environment(config_options, for_evaluation=True)
        model.set_env(env)
        
        # Update experiment status
        experiment_id = Path(experiment_path).name
        self._update_resume_status(experiment_id, starting_steps, additional_steps)
        
        # Create callbacks
        callbacks = self._create_resume_callbacks(experiment_path, eval_env, starting_steps)
        
        try:
            print("-" * 50)
            print("RESUMING TRAINING...")
            print("-" * 50)
            
            model.learn(
                total_timesteps=additional_steps,
                callback=callbacks,
                reset_num_timesteps=False,
                progress_bar=True
            )
            
            # Save final model with updated step count
            final_step_count = starting_steps + additional_steps
            final_model_path = Path(experiment_path) / "models" / f"final_model_resumed_{final_step_count}_steps"
            model.save(str(final_model_path))
            print(f"Resumed training completed. Final model saved to: {final_model_path}")
            
            # Update experiment status
            self.experiment_manager.update_experiment_status(experiment_id, "completed", {
                "resumed_from_steps": starting_steps,
                "final_steps": final_step_count,
                "additional_steps_trained": additional_steps
            })
            
        except Exception as e:
            print(f"Resume training failed: {e}")
            self.experiment_manager.update_experiment_status(experiment_id, "failed", {
                "resumed_from_steps": starting_steps,
                "error": str(e),
                "resume_attempt": True
            })
            raise
        
        return model, experiment_path
    
    @abstractmethod
    def _create_model(self, env, experiment_path):
        """Create algorithm-specific model - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _load_model(self, checkpoint_path):
        """Load algorithm-specific model - must be implemented by subclasses"""
        pass

    def _create_environment(self, options, render_mode=None, for_evaluation=False):
        """Create environment with appropriate settings"""
        k = getattr(self.cfg.experiment, 'num_envs', 1)
        env_import = self.cfg.experiment.env_import
        
        if render_mode is None:
            render_mode = "rgb_array"
        
        LV_params = {  # patient #1 ESS params (index 7): 6060.60606060606,7575.75757575758,1.e-09,27272.7272727273
            'init_counts': 0.4 * np.array([6060.60606060606, 7575.75757575758, 1.e-09], dtype=np.float64),  # T+, TP, T- (scaled to 10% as in paper)
            'growth_rates': np.array([0.27726, 0.34657, 0.66542], dtype=np.float64),
            'carrying_capacities': np.array([-1, 10000, 10000], dtype=np.float64),  # T+ depends on TP (1.5*TP; set in reset), TP (determines capacity of no-treatment), T-
            'tp_cap_on_treatment': np.array([100], dtype=np.float64),
            'competition_matrix': np.array([
                [1.0, 0.7, 0.8],  # T+ vs T+, TP, T-
                [0.4, 1.0, 0.5],  # TP vs T+, TP, T-
                [0.6, 0.9, 1.0]   # T- vs T+, TP, T-
            ], dtype=np.float64),
            'init_psa': 0.4 * np.array([27272.7272727273], dtype=np.float64),
        }  # patient #1 alpha params (index 7): 0.7,0.8,0.4,0.5,0.6,0.9

        env = ActionCoupledWrapper(
            env_fn=lambda render_mode=render_mode: gym.make(
                env_import,
                LV_params=LV_params,
                render_mode=render_mode,
                k=k),
            k=k,
            render_mode=render_mode if not for_evaluation else None,
            options=options,
            cfg=self.cfg.environment,
        )
        
        # Add normalization wrapper
        if hasattr(self.cfg, 'normalization') and self.cfg.normalization.enabled:
            env = NormalizationWrapper(
                env, 
                norm_obs=getattr(self.cfg.normalization, 'norm_obs', True),
                norm_reward=getattr(self.cfg.normalization, 'norm_reward', True)
            )
        
        if for_evaluation:
            env = Monitor(env)
        
        # Reset with options if provided
        # if options is not None:
        #     env.reset(options=options)
        # else:
        #     env.reset()
        env.reset()
        
        return env
    
    def _run_test_episodes(self, model, env, num_episodes, options=None, fixed_policy=None):
        """Run test episodes and collect results"""
        episode_rewards = []
        episode_steps = []
        
        for episode in range(num_episodes):
            # Reset with options for each episode
            if options:
                obs = env.reset(options=options)
            else:
                obs = env.reset()
                
            if isinstance(obs, tuple):
                obs = obs[0]
            total_reward = 0
            steps = 0
            
            print(f'Started episode {episode + 1}')
            while True:
                if fixed_policy:
                    if fixed_policy == 'MTD':
                        action = 1
                    if fixed_policy == 'Adaptive':
                        pop_norm = obs[-1]
                        if pop_norm <= 0.5:
                            action = 0
                        elif pop_norm >= 1.0:
                            action = 1
                    if fixed_policy == 'Optimal':
                        pop_norm = obs[-1]
                        if pop_norm < 1.16:
                            action = 0
                        elif pop_norm >= 1.16:
                            action = 1
                else:
                    action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                env.render()
                
                done = done or truncated
                total_reward += reward
                steps += 1
                
                if done:
                    print(f"Episode {episode + 1}: {steps} steps, reward: {total_reward}")
                    episode_rewards.append(total_reward)
                    episode_steps.append(steps)
                    break
        
        return episode_rewards, episode_steps
    
    def _compare_test_episodes(self, model, env, num_episodes, options=None):
        """Run test episodes and collect results"""
        episode_rewards = []
        episode_steps = []
        episode_scores = []
        
        for episode in range(num_episodes):
            # Reset with options for each episode
            if options:
                obs, info = env.reset(options=options)
            else:
                obs, info = env.reset()

            # Test trained model
            total_reward = 0
            steps = 0
            
            print(f'Started episode {episode + 1}')

            while True:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                # if episode == 1:
                #     print('action:', action, 'counts:', info['infos'][0]['counts'])
                env.render()
                
                done = done or truncated
                total_reward += reward
                steps += 1
                
                if done:
                    print(f"Episode {episode + 1}: {steps} steps, reward: {total_reward}")
                    episode_rewards.append(total_reward)
                    episode_steps.append(steps)
                    break
        
            
            # Test the optimal policy for the same initial state
            total_reward = 0
            steps = 0
            THRESHOLD = 1.16
            obs, info = env.reset(options=options)
            while True:
                pop_norm = obs[-1] + 0.6
                if pop_norm < THRESHOLD:
                    action = 0
                elif pop_norm >= THRESHOLD:
                    action = 1
                obs, reward, done, truncated, info = env.step(action)
                env.render()
                
                done = done or truncated
                total_reward += reward
                steps += 1
                
                if done:
                    print(f"Threshold policy ({THRESHOLD}) done in: {steps} steps, reward: {total_reward}")
                    score = episode_rewards[-1] / total_reward
                    episode_scores.append(score)
                    break
        print('Score compared to threshold policy', episode_scores)
        print('Average Episode Score:', np.mean(episode_scores))
        
        return episode_rewards, episode_steps, episode_scores

    
    def _create_callbacks(self, experiment_path, eval_env):
        """Create training callbacks with enhanced TensorBoard logging"""
        
        # Existing callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=self.cfg.training.save_freq,
            save_path=str(Path(experiment_path) / "models"),
            name_prefix="checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=True,
            verbose=0
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(Path(experiment_path) / "models"),
            log_path=str(Path(experiment_path) / "logs"),
            eval_freq=self.cfg.training.eval_freq,
            deterministic=getattr(self.cfg.training, 'eval_deterministic', True),
            render=False,
            verbose=0,
            n_eval_episodes=getattr(self.cfg.training, 'n_eval_episodes', 5),
        )
        
        # New custom TensorBoard callback
        # Check if logging configuration exists, otherwise use defaults
        log_frequency = getattr(self.cfg.logging, 'log_frequency', 1000) if hasattr(self.cfg, 'logging') else 1000
        track_gradients = getattr(self.cfg.logging, 'track_gradients', False) if hasattr(self.cfg, 'logging') else False
        track_data_stats = getattr(self.cfg.logging, 'track_data_stats', True) if hasattr(self.cfg, 'logging') else True
        
        custom_tb_callback = CustomTensorboardCallback(
            log_frequency=log_frequency,
            track_gradients=track_gradients,
            track_data_stats=track_data_stats,
            verbose=0
        )
        
        # Return all callbacks
        callbacks = [checkpoint_callback, eval_callback, custom_tb_callback]
        
        return callbacks
    
    def _extract_step_count(self, checkpoint_path, model):
        """Extract step count from checkpoint path or model"""
        filename = os.path.basename(checkpoint_path)
        
        step_patterns = [
            r'checkpoint_(\d+)_steps',
            r'final_model_resumed_(\d+)_steps',
            r'_(\d+)_steps'
        ]
        
        for pattern in step_patterns:
            match = re.search(pattern, filename)
            if match:
                steps = int(match.group(1))
                print(f"Extracted {steps} steps from filename: {filename}")
                return steps
        
        if hasattr(model, 'num_timesteps'):
            steps = model.num_timesteps
            print(f"Using model's internal timestep counter: {steps}")
            return steps
        
        print("Warning: Could not determine starting step count, assuming 0")
        return 0
    
    def _extract_experiment_path_from_model(self, model_path):
        """Extract experiment path from model file path"""
        model_path = Path(model_path).resolve()
        return str(model_path.parent.parent)
    
    def _update_resume_status(self, experiment_id, starting_steps, additional_steps):
        """Update experiment registry for resumed training"""
        try:
            self.experiment_manager.update_experiment_status(experiment_id, "resuming")
        except Exception as e:
            print(f"Warning: Could not update experiment registry: {e}")
    
    def _print_test_summary(self, episode_rewards, episode_steps):
        """Print testing summary statistics"""
        print("-" * 50)
        print("TESTING SUMMARY:")
        print(f"Average reward: {sum(episode_rewards)/len(episode_rewards):.2f}")
        print(f"Average steps: {sum(episode_steps)/len(episode_steps):.2f}")
        print(f"Best reward: {max(episode_rewards):.2f}")
        print(f"Worst reward: {min(episode_rewards):.2f}")
    
    def _create_resume_callbacks(self, experiment_path, eval_env, starting_steps):
        """Create callbacks for resumed training"""
        class ContinuedCheckpointCallback(CheckpointCallback):
            def __init__(self, *args, starting_steps=0, **kwargs):
                super().__init__(*args, **kwargs)
                self.starting_steps = starting_steps
            
            def _on_step(self) -> bool:
                if self.n_calls % self.save_freq == 0:
                    total_steps = self.starting_steps + self.num_timesteps
                    path = os.path.join(self.save_path, f"checkpoint_{total_steps}_steps")
                    self.model.save(path)
                    if self.verbose >= 2:
                        print(f"Saving resumed checkpoint to {path}")
                return True
        
        checkpoint_callback = ContinuedCheckpointCallback(
            save_freq=self.cfg.training.save_freq,
            save_path=str(Path(experiment_path) / "models"),
            name_prefix="checkpoint",
            starting_steps=starting_steps,
            verbose=1
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(Path(experiment_path) / "models"),
            log_path=str(Path(experiment_path) / "logs"),
            eval_freq=self.cfg.training.eval_freq,
            deterministic=True,
            render=False,
            verbose=1,
            n_eval_episodes=5,
        )
        
        return [checkpoint_callback, eval_callback]