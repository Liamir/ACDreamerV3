"""
PPO Trainer Module
PPO-specific implementation inheriting from BaseTrainer
"""

from stable_baselines3 import PPO
from pathlib import Path
from .base_trainer import BaseTrainer


class PPOTrainer(BaseTrainer):
    """PPO Training Manager - inherits common functionality from BaseTrainer"""
    
    def _create_model(self, env, experiment_path):
        """Create PPO model with experiment-specific tensorboard logging"""
        tensorboard_path = Path(experiment_path) / "logs" / "tensorboard"
        tensorboard_path.mkdir(parents=True, exist_ok=True)
        
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=self.cfg.algorithm.hyperparameters.learning_rate,
            n_steps=self.cfg.algorithm.hyperparameters.n_steps,
            batch_size=self.cfg.algorithm.hyperparameters.batch_size,
            n_epochs=self.cfg.algorithm.hyperparameters.n_epochs,
            gamma=self.cfg.algorithm.hyperparameters.gamma,
            gae_lambda=self.cfg.algorithm.hyperparameters.gae_lambda,
            clip_range=self.cfg.algorithm.hyperparameters.clip_range,
            clip_range_vf=self.cfg.algorithm.hyperparameters.clip_range_vf,
            ent_coef=self.cfg.algorithm.hyperparameters.ent_coef,
            vf_coef=self.cfg.algorithm.hyperparameters.vf_coef,
            max_grad_norm=self.cfg.algorithm.hyperparameters.max_grad_norm,
            use_sde=self.cfg.algorithm.hyperparameters.use_sde,
            sde_sample_freq=self.cfg.algorithm.hyperparameters.sde_sample_freq,
            target_kl=self.cfg.algorithm.hyperparameters.target_kl,
            tensorboard_log=str(tensorboard_path),
            device='cpu',
        )
        
        return model
    
    def _load_model(self, checkpoint_path):
        """Load PPO model from checkpoint"""
        try:
            model = PPO.load(checkpoint_path)
            print(f"Successfully loaded PPO model from: {checkpoint_path}")
            return model
        except Exception as e:
            print(f"Error loading PPO checkpoint: {e}")
            return None