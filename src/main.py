"""
Main Entry Point for RL Training Pipeline
Simple orchestrator that coordinates all components
"""

import sys
from .utils.cli import create_argument_parser, load_config_with_cli_overrides, print_configuration_summary
from .trainers.ppo_trainer import PPOTrainer
from .trainers.dqn_trainer import DQNTrainer
from .core.experiment import ExperimentManager


def create_trainer(cfg):
    """Factory function to create the appropriate trainer based on algorithm"""
    algorithm = cfg.algorithm.name.lower()
    
    if algorithm == "ppo":
        return PPOTrainer(cfg)
    elif algorithm == "dqn":
        return DQNTrainer(cfg)
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
    
    trainer = create_trainer(cfg)
    trainer.train()
    
    print("✅ Training completed successfully!")


def handle_test_command(cfg, args):
    """Handle test command"""
    print(f"\n🧪 Starting {cfg.algorithm.name} Testing...")
    
    trainer = create_trainer(cfg)
    trainer.test(model_path=args.model_path)
    
    print("✅ Testing completed!")


def handle_resume_command(cfg, args):
    """Handle resume command"""
    print(f"\n🔄 Resuming {cfg.algorithm.name} Training...")
    
    trainer = create_trainer(cfg)
    trainer.resume(model_path=args.model_path)
    
    print("✅ Resume training completed!")


def handle_list_command(cfg, args):
    """Handle list command"""
    print("\n📋 Listing Experiments...")
    
    experiment_manager = ExperimentManager(cfg.experiment.save_path)
    experiment_manager.list_experiments()


if __name__ == "__main__":
    main()