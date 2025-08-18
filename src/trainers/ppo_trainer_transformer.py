"""
PPO Trainer Module
PPO-specific implementation inheriting from BaseTrainer
"""

from stable_baselines3 import PPO
from pathlib import Path
from .base_trainer import BaseTrainer
from ..models.set_transformer import SetTransformerExtractor


class PPOTrainer(BaseTrainer):
    """PPO Training Manager - inherits common functionality from BaseTrainer"""
    
    def _create_model(self, env, experiment_path):
        """Create PPO model with experiment-specific tensorboard logging"""
        tensorboard_path = Path(experiment_path) / "logs" / "tensorboard"
        tensorboard_path.mkdir(parents=True, exist_ok=True)

        if hasattr(env, "k"):
            num_envs = env.k
        else:
            num_envs = self.cfg.experiment.num_envs
            print(f"Warning: Could not detect num_envs from environment, using config value: {num_envs}")

        set_transformer_config = {
            "num_envs": num_envs,
            "features_dim": getattr(self.cfg.algorithm.hyperparameters, 'features_dim', 128),
            "d_model": getattr(self.cfg.algorithm.hyperparameters, 'd_model', 64),
            "num_heads": getattr(self.cfg.algorithm.hyperparameters, 'num_heads', 4),
            "num_layers": getattr(self.cfg.algorithm.hyperparameters, 'num_layers', 2),
            "dropout": getattr(self.cfg.algorithm.hyperparameters, 'dropout', 0.1),
        }

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
            policy_kwargs={
                "features_extractor_class": SetTransformerExtractor,
                "features_extractor_kwargs": set_transformer_config,
            }
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