"""
SAC Trainer Module
SAC-specific implementation inheriting from BaseTrainer
"""

from stable_baselines3 import SAC
from pathlib import Path
from .base_trainer import BaseTrainer


class SACTrainer(BaseTrainer):
    """SAC Training Manager - inherits common functionality from BaseTrainer"""
    
    def _create_model(self, env, experiment_path):
        """Create SAC model with experiment-specific tensorboard logging"""
        tensorboard_path = Path(experiment_path) / "logs" / "tensorboard"
        tensorboard_path.mkdir(parents=True, exist_ok=True)

        # Get SAC-specific hyperparameters
        hp = self.cfg.algorithm.hyperparameters
        
        # SAC uses continuous action spaces, so we need to ensure the environment is compatible
        if not hasattr(env.action_space, 'low') or not hasattr(env.action_space, 'high'):
            raise ValueError("SAC requires continuous action spaces (Box). Environment action space is: {}".format(type(env.action_space)))

        model = SAC(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=hp.learning_rate,
            buffer_size=getattr(hp, 'buffer_size', 1000000),
            learning_starts=getattr(hp, 'learning_starts', 100),
            batch_size=getattr(hp, 'batch_size', 256),
            tau=getattr(hp, 'tau', 0.005),
            gamma=getattr(hp, 'gamma', 0.99),
            train_freq=getattr(hp, 'train_freq', 1),
            gradient_steps=getattr(hp, 'gradient_steps', 1),
            action_noise=None,  # SAC uses entropy regularization instead of action noise
            ent_coef=getattr(hp, 'ent_coef', 'auto'),  # 'auto' for automatic entropy tuning
            target_update_interval=getattr(hp, 'target_update_interval', 1),
            target_entropy=getattr(hp, 'target_entropy', 'auto'),  # 'auto' for automatic target entropy
            use_sde=getattr(hp, 'use_sde', False),
            sde_sample_freq=getattr(hp, 'sde_sample_freq', -1),
            use_sde_at_warmup=getattr(hp, 'use_sde_at_warmup', False),
            tensorboard_log=str(tensorboard_path),
            device='cpu',
            policy_kwargs={
                "net_arch": getattr(hp, 'net_arch', [256, 256]),  # Default: 2 hidden layers with 256 units
                "n_critics": getattr(hp, 'n_critics', 2),  # Number of critic networks (default: 2 for SAC)
            }
        )
        
        return model
    
    def _load_model(self, checkpoint_path):
        """Load SAC model from checkpoint"""
        try:
            model = SAC.load(checkpoint_path)
            print(f"Successfully loaded SAC model from: {checkpoint_path}")
            return model
        except Exception as e:
            print(f"Error loading SAC checkpoint: {e}")
            return None