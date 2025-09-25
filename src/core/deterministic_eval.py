"""
Deterministic Evaluation Module for RL Training
Provides fixed evaluation sets to ensure consistent model comparison
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Logger
import gymnasium as gym
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json


class FixedEvalStates:
    """Manages a fixed set of evaluation states for consistent model comparison"""
    
    def __init__(self, cfg, num_states: int = 10):
        self.cfg = cfg
        self.num_states = num_states
        self.carrying_capacity = cfg.environment.params.carrying_capacity
        self.fixed_states = self._generate_fixed_states()
        
    def _generate_fixed_states(self) -> List[Dict[str, float]]:
        """Generate a diverse set of fixed initial states"""
        np.random.seed(42)  # Fixed seed for reproducibility
        
        states = []
        
        # Define different scenarios to ensure good coverage
        scenarios = [
            # Low population scenarios
            {"s_frac": 0.05, "r_frac": 0.05},   # Very low both
            {"s_frac": 0.1, "r_frac": 0.05},    # Low S, very low R
            {"s_frac": 0.05, "r_frac": 0.1},    # Very low S, low R
            
            # Medium population scenarios  
            {"s_frac": 0.3, "r_frac": 0.2},     # Medium mixed
            {"s_frac": 0.4, "r_frac": 0.1},     # Medium S, low R
            {"s_frac": 0.2, "r_frac": 0.3},     # Low S, medium R
            
            # High population scenarios
            {"s_frac": 0.1, "r_frac": 0.001},     # High S, medium R
            {"s_frac": 0.001, "r_frac": 0.1},     # Medium S, high R
            
            # Extreme scenarios
            {"s_frac": 0.7, "r_frac": 0.0001},     # Very high S, low R
            {"s_frac": 0.0001, "r_frac": 0.7},     # Low S, very high R
        ]
        
        for scenario in scenarios:
            s_count = scenario["s_frac"] * self.carrying_capacity
            r_count = scenario["r_frac"] * self.carrying_capacity
            
            state = {
                "s_counts": s_count,
                "r_counts": r_count,
            }
            states.append(state)
            
        return states[:self.num_states]
    
    def get_states(self) -> List[Dict[str, float]]:
        """Return the fixed evaluation states"""
        return self.fixed_states
    
    def save_states(self, filepath: str):
        """Save fixed states to file for reproducibility"""
        with open(filepath, 'w') as f:
            json.dump(self.fixed_states, f, indent=2)
    
    def load_states(self, filepath: str):
        """Load fixed states from file"""
        with open(filepath, 'r') as f:
            self.fixed_states = json.load(f)


class DeterministicEvaluator:
    """Handles deterministic evaluation using fixed initial states"""
    
    def __init__(self, cfg, eval_env, fixed_states: FixedEvalStates, 
                 episodes_per_state: int = 3):
        self.cfg = cfg
        self.eval_env = eval_env
        self.fixed_states = fixed_states
        self.episodes_per_state = episodes_per_state
        
    def evaluate_model(self, model, deterministic: bool = True) -> Dict[str, Any]:
        """
        Evaluate model on fixed states and return comprehensive statistics
        
        Returns:
            Dictionary containing mean reward, std, per-state results, etc.
        """
        all_rewards = []
        all_episode_lengths = []
        state_results = []
        
        for i, state in enumerate(self.fixed_states.get_states()):
            state_rewards = []
            state_lengths = []
            
            for episode in range(self.episodes_per_state):
                # Create options dict for reset
                options = {
                    'init_state': {
                        's_counts': [state['s_counts']],
                        'r_counts': [state['r_counts']]
                    }
                }
                
                # Run single episode
                obs, info = self.eval_env.reset(options=options)
                if isinstance(obs, tuple):
                    obs = obs[0]
                
                total_reward = 0
                steps = 0
                done = False
                
                while not done:
                    action, _ = model.predict(obs, deterministic=deterministic)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                    steps += 1
                
                state_rewards.append(total_reward)
                state_lengths.append(steps)
                all_rewards.append(total_reward)
                all_episode_lengths.append(steps)
            
            # Store per-state statistics
            state_results.append({
                'state_index': i,
                'state': state,
                'mean_reward': np.mean(state_rewards),
                'std_reward': np.std(state_rewards),
                'mean_length': np.mean(state_lengths),
                'rewards': state_rewards
            })
        
        # Calculate overall statistics
        results = {
            'mean_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'mean_episode_length': np.mean(all_episode_lengths),
            'std_episode_length': np.std(all_episode_lengths),
            'min_reward': np.min(all_rewards),
            'max_reward': np.max(all_rewards),
            'total_episodes': len(all_rewards),
            'state_results': state_results,
            'all_rewards': all_rewards
        }
        
        return results
    
    def compare_models(self, model1, model2, model1_name: str = "Model 1", 
                      model2_name: str = "Model 2") -> Dict[str, Any]:
        """Compare two models using statistical testing"""
        from scipy import stats
        
        results1 = self.evaluate_model(model1)
        results2 = self.evaluate_model(model2)
        
        rewards1 = results1['all_rewards']
        rewards2 = results2['all_rewards']
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(rewards1, rewards2)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(rewards1) - 1) * results1['std_reward']**2 + 
                             (len(rewards2) - 1) * results2['std_reward']**2) / 
                            (len(rewards1) + len(rewards2) - 2))
        cohens_d = (results1['mean_reward'] - results2['mean_reward']) / pooled_std
        
        comparison = {
            'model1_name': model1_name,
            'model2_name': model2_name,
            'model1_results': results1,
            'model2_results': results2,
            'statistical_test': {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'cohens_d': cohens_d,
                'effect_size': 'small' if abs(cohens_d) < 0.5 else ('medium' if abs(cohens_d) < 0.8 else 'large')
            }
        }
        
        return comparison
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted evaluation results"""
        print("=" * 60)
        print("DETERMINISTIC EVALUATION RESULTS")
        print("=" * 60)
        print(f"Mean Reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
        print(f"Episode Length: {results['mean_episode_length']:.1f} ± {results['std_episode_length']:.1f}")
        print(f"Reward Range: [{results['min_reward']:.3f}, {results['max_reward']:.3f}]")
        print(f"Total Episodes: {results['total_episodes']}")
        print("\nPer-State Results:")
        print("-" * 40)
        
        for state_result in results['state_results']:
            state = state_result['state']
            print(f"State {state_result['state_index']}: "
                  f"S={state['s_counts']:.1f}, R={state['r_counts']:.1f} | "
                  f"Reward: {state_result['mean_reward']:.3f} ± {state_result['std_reward']:.3f}")


class DeterministicEvalCallback(BaseCallback):
    """Callback that performs deterministic evaluation during training"""
    
    def __init__(self, eval_env, fixed_states: FixedEvalStates, eval_freq: int = 10000,
                 episodes_per_state: int = 3, save_path: Optional[str] = None,
                 verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.evaluator = DeterministicEvaluator(
            cfg=None,  # Not needed for callback
            eval_env=eval_env,
            fixed_states=fixed_states,
            episodes_per_state=episodes_per_state
        )
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        self.eval_results_history = []
        
    def _init_callback(self) -> None:
        if self.save_path is not None:
            Path(self.save_path).mkdir(parents=True, exist_ok=True)
    
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            
            # Evaluate model
            results = self.evaluator.evaluate_model(self.model, deterministic=True)
            self.eval_results_history.append(results)
            
            # Log to tensorboard
            mean_reward = results['mean_reward']
            std_reward = results['std_reward']
            mean_length = results['mean_episode_length']
            
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/std_reward", std_reward)
            self.logger.record("eval/mean_ep_length", mean_length)
            
            # Log per-state results
            for state_result in results['state_results']:
                state_idx = state_result['state_index']
                self.logger.record(f"eval/state_{state_idx}_reward", state_result['mean_reward'])
            
            if self.verbose >= 1:
                print(f"Eval at step {self.num_timesteps}: "
                      f"mean_reward={mean_reward:.3f} ± {std_reward:.3f}")
            
            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.save_path is not None:
                    best_model_path = Path(self.save_path) / "best_deterministic_model"
                    self.model.save(str(best_model_path))
                    if self.verbose >= 1:
                        print(f"New best model saved with reward {mean_reward:.3f}")
        
        return True
    
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """Return history of all evaluations"""
        return self.eval_results_history


def create_deterministic_eval_callback(cfg, eval_env, experiment_path: str, 
                                     eval_freq: int = 10000, num_eval_states: int = 10,
                                     episodes_per_state: int = 3) -> DeterministicEvalCallback:
    """
    Convenience function to create deterministic evaluation callback
    
    Args:
        cfg: Configuration object
        eval_env: Evaluation environment
        experiment_path: Path to save results and models
        eval_freq: How often to evaluate (in timesteps)
        num_eval_states: Number of fixed states to use
        episodes_per_state: Episodes per state for evaluation
    
    Returns:
        Configured DeterministicEvalCallback
    """
    # Create fixed states
    fixed_states = FixedEvalStates(cfg, num_states=num_eval_states)
    
    # Save fixed states for reproducibility
    states_path = Path(experiment_path) / "fixed_eval_states.json"
    fixed_states.save_states(str(states_path))
    
    # Create callback
    callback = DeterministicEvalCallback(
        eval_env=eval_env,
        fixed_states=fixed_states,
        eval_freq=eval_freq,
        episodes_per_state=episodes_per_state,
        save_path=str(Path(experiment_path) / "models"),
        verbose=1
    )
    
    return callback