"""
PPO Trainer Module
Handles PPO-specific training, testing, and resume functionality
"""

import os
import re
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from pathlib import Path

# Import custom environments and wrappers
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import custom_envs
from custom_envs.action_coupled_wrapper_v3 import ActionCoupledWrapper

from ..core.experiment import ExperimentManager, print_env_info


class PPOTrainer:
    """PPO Training Manager"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.experiment_manager = ExperimentManager(cfg.experiment.save_path)
    
    def train(self, init_ranges=None):
        """Train PPO model with checkpoints and evaluation"""
        print(f"Starting PPO training for {self.cfg.training.total_timesteps} steps")
        
        # Create experiment structure
        experiment_path, experiment_id = self.experiment_manager.create_experiment_folder(self.cfg)
        self.experiment_manager.register_experiment(experiment_path, experiment_id, self.cfg)
        self.experiment_manager.save_experiment_config(self.cfg, experiment_path)
        
        print(f"Experiment ID: {experiment_id}")
        print(f"Experiment path: {experiment_path}")
        
        try:
            # Create environments
            env = self._create_environment(init_ranges)
            eval_env = self._create_environment(init_ranges, for_evaluation=True)
            
            print_env_info(env, self.cfg)
            
            # Create PPO model
            model = self._create_ppo_model(env, experiment_path)
            
            # Setup callbacks
            callbacks = self._create_callbacks(experiment_path, eval_env)
            
            # Train the model
            print(f"Starting training...")
            print(f"Logs and checkpoints will be saved to: {experiment_path}")
            
            model.learn(
                total_timesteps=self.cfg.training.total_timesteps,
                callback=callbacks,
                progress_bar=True
            )
            
            # Save final model
            final_model_path = Path(experiment_path) / "models" / "final_model"
            model.save(str(final_model_path))
            print(f"Final model saved to: {final_model_path}")
            
            self.experiment_manager.update_experiment_status(experiment_id, "completed")
            
        except Exception as e:
            print(f"Training failed: {e}")
            self.experiment_manager.update_experiment_status(experiment_id, "failed", {"error": str(e)})
            raise
        
        return model
    
    def test(self, model_path=None, init_ranges=None):
        """Test a trained PPO model"""

        num_episodes = self.cfg.evaluation.episodes

        # Determine model path
        if model_path:
            checkpoint_path = model_path
            print(f"Using provided model path: {checkpoint_path}")
        else:
            checkpoint_path = self.experiment_manager.find_model_from_config(self.cfg)
            if not checkpoint_path:
                print("Could not find model. Please provide model_path or ensure experiment exists.")
                return
        
        # Load model
        model = self._load_model(checkpoint_path)
        if model is None:
            return
        
        # Create test environment
        env = self._create_environment(init_ranges, render_mode="human")
        
        print(f'Testing model from: {checkpoint_path}')
        print(f'Environment: {self.cfg.experiment.env_import}')
        print(f'Number of episodes: {num_episodes}')
        print("-" * 50)
        
        # Test episodes
        episode_rewards, episode_steps = self._run_test_episodes(model, env, num_episodes)
        
        # Print summary
        self._print_test_summary(episode_rewards, episode_steps)
        
        env.close()
        return episode_rewards, episode_steps
    
    def resume(self, model_path=None, additional_steps=None, model_type="best", init_ranges=None):
        """Resume training from a checkpoint"""
        # Determine additional steps
        if additional_steps is None:
            additional_steps = self.cfg.training.total_timesteps
        
        # Find the model to resume from
        if model_path:
            checkpoint_path = model_path
            experiment_path = self._extract_experiment_path_from_model(model_path)
        else:
            checkpoint_path = self.experiment_manager.find_model_from_config(self.cfg, model_type)
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
        
        # Create environments
        env = self._create_environment(init_ranges)
        eval_env = self._create_environment(init_ranges, for_evaluation=True)
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
    
    def _create_environment(self, init_ranges=None, render_mode=None, for_evaluation=False):
        """Create environment with appropriate settings"""
        k = self.cfg.experiment.num_envs
        env_import = self.cfg.experiment.env_import
        
        if render_mode is None:
            render_mode = "rgb_array"
        
        env = ActionCoupledWrapper(
            env_fn=lambda render_mode=render_mode: gym.make(env_import, render_mode=render_mode),
            k=k,
            render_mode=render_mode if not for_evaluation else None,
            init_ranges=init_ranges
        )
        
        if for_evaluation:
            env = Monitor(env)
        
        return env
    
    def _create_ppo_model(self, env, experiment_path):
        """Create PPO model with experiment-specific tensorboard logging"""
        tensorboard_path = Path(experiment_path) / "logs" / "tensorboard"
        tensorboard_path.mkdir(parents=True, exist_ok=True)
        
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=self.cfg.algorithm.hyperparameters.learning_rate,
            n_steps=self.cfg.algorithm.hyperparameters.n_steps,
            batch_size=self.cfg.algorithm.hyperparameters.batch_size,
            n_epochs=self.cfg.algorithm.hyperparameters.n_epochs,
            gamma=self.cfg.algorithm.hyperparameters.gamma,
            gae_lambda=self.cfg.algorithm.hyperparameters.gae_lambda,
            clip_range=self.cfg.algorithm.hyperparameters.clip_range,
            tensorboard_log=str(tensorboard_path)
        )
        
        return model
    
    def _create_callbacks(self, experiment_path, eval_env):
        """Create training callbacks"""
        checkpoint_callback = CheckpointCallback(
            save_freq=self.cfg.training.save_freq,
            save_path=str(Path(experiment_path) / "models"),
            name_prefix="checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=True,
            verbose=2
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
            verbose=2
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
    
    def _load_model(self, checkpoint_path):
        """Load model from checkpoint"""
        try:
            model = PPO.load(checkpoint_path)
            print(f"Successfully loaded model from: {checkpoint_path}")
            return model
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None
    
    def _run_test_episodes(self, model, env, num_episodes):
        """Run test episodes and collect results"""
        episode_rewards = []
        episode_steps = []
        
        for episode in range(num_episodes):
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            total_reward = 0
            steps = 0
            
            print(f'Started episode {episode + 1}')
            while True:
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
    
    def _print_test_summary(self, episode_rewards, episode_steps):
        """Print testing summary statistics"""
        print("-" * 50)
        print("TESTING SUMMARY:")
        print(f"Average reward: {sum(episode_rewards)/len(episode_rewards):.2f}")
        print(f"Average steps: {sum(episode_steps)/len(episode_steps):.2f}")
        print(f"Best reward: {max(episode_rewards):.2f}")
        print(f"Worst reward: {min(episode_rewards):.2f}")
    
    def _extract_step_count(self, checkpoint_path, model):
        """Extract step count from checkpoint path or model"""
        filename = os.path.basename(checkpoint_path)
        
        # Look for patterns like "checkpoint_50000_steps"
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
        
        # Fall back to model's internal counter
        if hasattr(model, 'num_timesteps'):
            steps = model.num_timesteps
            print(f"Using model's internal timestep counter: {steps}")
            return steps
        
        print("Warning: Could not determine starting step count, assuming 0")
        return 0
    
    def _extract_experiment_path_from_model(self, model_path):
        """Extract experiment path from model file path"""
        model_path = Path(model_path).resolve()
        # Go up two levels: from model_file.zip -> models -> experiment_folder
        return str(model_path.parent.parent)
    
    def _update_resume_status(self, experiment_id, starting_steps, additional_steps):
        """Update experiment registry for resumed training"""
        try:
            self.experiment_manager.update_experiment_status(experiment_id, "resuming")
            # Could add more detailed resume info here
        except Exception as e:
            print(f"Warning: Could not update experiment registry: {e}")