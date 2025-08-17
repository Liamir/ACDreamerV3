"""
Main Entry Point for RL Training Pipeline
Simple orchestrator that coordinates all components
"""

import sys
import copy
import random
import json
import csv
import math
from pathlib import Path
from datetime import datetime

from .utils.cli import create_argument_parser, load_config_with_cli_overrides, print_configuration_summary
# from .trainers.ppo_trainer import PPOTrainer
from .trainers.ppo_trainer_transformer import PPOTrainer
from .trainers.sac_trainer import SACTrainer
from .trainers.dqn_trainer import DQNTrainer
from .core.experiment import ExperimentManager


def create_trainer(cfg):
    """Factory function to create the appropriate trainer based on algorithm"""
    algorithm = cfg.algorithm.name.lower()
    
    if algorithm == "ppo":
        return PPOTrainer(cfg)
    elif algorithm == "dqn":
        return DQNTrainer(cfg)
    elif algorithm == "sac":
        return SACTrainer(cfg)
    else:
        supported_algorithms = ["ppo", "dqn"]
        raise ValueError(f"Unsupported algorithm: {algorithm}. Supported: {supported_algorithms}")


def main():
    """Main function that orchestrates the training pipeline"""
    
    print("=" * 60)
    print("RL TRAINING PIPELINE")
    print("=" * 60)
    
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Load configuration with CLI overrides
    try:
        config_manager = load_config_with_cli_overrides(args.config, args)
        cfg = config_manager.config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Print configuration summary
    print_configuration_summary(cfg, args.command)
    
    # Route to appropriate command handler
    try:
        if args.command == "train":
            handle_train_command(cfg, args)
        elif args.command == "test":
            handle_test_command(cfg, args)
        elif args.command == "resume":
            handle_resume_command(cfg, args)
        elif args.command == "tune":
            handle_tune_command(cfg, args)
        elif args.command == "list":
            handle_list_command(cfg, args)
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def handle_train_command(cfg, args):
    """Handle train command"""
    print(f"\n🚀 Starting {cfg.algorithm.name} Training...")
    
    import time
    start_time = time.time()
    trainer = create_trainer(cfg)
    trainer.train()
    training_time = time.time() - start_time
    
    print("✅ Training completed successfully!")
    print(f"Training time: {training_time:.1f}s")


def handle_tune_command(cfg, args):
    """Handle tune command"""
    if not cfg.algorithm.tuning.enabled:
        print("Tuning is disabled in config.")
        return

    num_trials = cfg.algorithm.tuning.num_trials
    search_space = cfg.algorithm.tuning.search_space

    # Create a unique folder for this tuning run    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    k = cfg.experiment.num_envs
    ac_str = f'{k}ac_' if k > 1 else ''
    run_name = cfg.experiment.name
    env_name = cfg.experiment.env_import.split('-')[0].lower()
    tuning_folder = Path(f"{cfg.algorithm.name.lower()}_{ac_str}{env_name}_{run_name}_{timestamp}")
    
    base_path = cfg.experiment.save_path
    tuning_path = base_path / tuning_folder
    tuning_path.mkdir(parents=True, exist_ok=True)
    csv_file = tuning_path / "tuning_log.csv"
    fieldnames = ['trial', 'experiment_name', 'timestamp'] + list(search_space.keys())

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(num_trials):
            print("=" * 60)
            print(f"Trial {i + 1}/{num_trials}")
            print("=" * 60)

            # Deep copy base config
            trial_cfg = copy.deepcopy(cfg)

            # Sample hyperparameters
            sampled_params = {}
            for param, (scale, low, high) in search_space.items():
                if scale == 'linear':
                    if isinstance(low, int) and isinstance(high, int):
                        value = random.randint(low, high)
                    else:
                        value = random.uniform(float(low), float(high))
                
                elif scale == 'log':
                    value = 10 ** random.uniform(math.log10(low), math.log10(high))

                trial_cfg.algorithm.hyperparameters[param] = value
                sampled_params[param] = value

            # Unique experiment name and path for the trial
            trial_name = f"trial_{i + 1}"
            trial_cfg.experiment.name = trial_name
            trial_cfg.experiment.save_path = str(tuning_path / trial_name)

            # Create and train
            trainer = create_trainer(trial_cfg)
            trainer.train()

            # Write log row
            log_entry = {
                "trial": i + 1,
                "experiment_name": trial_name,
                "timestamp": datetime.now().isoformat(),
                **sampled_params,
            }
            writer.writerow(log_entry)

    print("✅ Tuning completed successfully!")
    print(f"📁 All trials saved under: {tuning_path}")
    print(f"📄 Tuning log saved to: {csv_file}")


def handle_test_command(cfg, args):
    """Handle test command"""
    print(f"\n🧪 Starting {cfg.algorithm.name} Testing...")
    
    trainer = create_trainer(cfg)
    try:
        trainer.test()
    except Exception as e:
        print(f"Testing failed: {e}")
        raise

    print("✅ Testing completed!")


def handle_resume_command(cfg, args):
    """Handle resume command"""
    print(f"\n🔄 Resuming {cfg.algorithm.name} Training...")
    
    trainer = create_trainer(cfg)
    trainer.resume()
    
    print("✅ Resume training completed!")


def handle_list_command(cfg, args):
    """Handle list command"""
    print("\n📋 Listing Experiments...")
    
    experiment_manager = ExperimentManager(cfg.experiment.save_path)
    experiment_manager.list_experiments()


if __name__ == "__main__":
    main()