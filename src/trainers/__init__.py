"""Training modules for different RL algorithms"""

# from .ppo_trainer import PPOTrainer
from .ppo_trainer_transformer import PPOTrainerTransformer
from .ppo_trainer_transformer import PPOTrainerTransformer
from .dqn_trainer import DQNTrainer

__all__ = ['PPOTrainerTransformer', 'PPOTrainer', 'DQNTrainer']