"""
DQN Trainer Module
DQN-specific implementation inheriting from BaseTrainer
"""

from stable_baselines3 import DQN
from pathlib import Path
from .base_trainer import BaseTrainer


class DQNTrainer(BaseTrainer):
    """DQN Training Manager - inherits common functionality from BaseTrainer"""
    
    def _create_model(self, env, experiment_path):
        """Create DQN model with experiment-specific tensorboard logging"""
        tensorboard_path = Path(experiment_path) / "logs" / "tensorboard"
        tensorboard_path.mkdir(parents=True, exist_ok=True)
        
        # Get DQN-specific hyperparameters
        hp = self.cfg.algorithm.hyperparameters
        
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=hp.learning_rate,
            buffer_size=getattr(hp, 'buffer_size', 50000),
            learning_starts=getattr(hp, 'learning_starts', 1000),
            batch_size=getattr(hp, 'batch_size', 128),
            target_update_interval=getattr(hp, 'target_update_interval', 600),
            train_freq=getattr(hp, 'train_freq', 4),
            gradient_steps=getattr(hp, 'gradient_steps', 1),
            exploration_fraction=getattr(hp, 'exploration_fraction', 0.3),
            exploration_initial_eps=getattr(hp, 'exploration_initial_eps', 1.0),
            exploration_final_eps=getattr(hp, 'exploration_final_eps', 0.02),
            gamma=getattr(hp, 'gamma', 0.99),
            tensorboard_log=str(tensorboard_path)
        )
        
        return model
    
    def _load_model(self, checkpoint_path):
        """Load DQN model from checkpoint"""
        try:
            model = DQN.load(checkpoint_path)
            print(f"Successfully loaded DQN model from: {checkpoint_path}")
            return model
        except Exception as e:
            print(f"Error loading DQN checkpoint: {e}")
            return None