import numpy as np
import matplotlib.pyplot as plt
from cell_group import ProstateCancerTherapyEnv

def main():
    # Define Lotka-Volterra parameters based on the paper
    LV_params = {  # patient #1 ESS params (index 7): 6060.60606060606,7575.75757575758,1.e-09,27272.7272727273
        'ess_counts': np.array([6060.60606060606, 7575.75757575758, 1.e-09], dtype=np.float64),  # T+, TP, T- (scaled to 10% as in paper)
        'growth_rates': np.array([0.27726, 0.34657, 0.66542], dtype=np.float64),
        'carrying_capacities': np.array([-1, 10000, 10000], dtype=np.float64),  # T+ depends on TP (1.5*TP; set in reset), TP (determines capacity of no-treatment), T-
        'tp_cap_on_treatment': np.array([100], dtype=np.float64),
        'competition_matrix': np.array([
            [1.0, 0.7, 0.8],  # T+ vs T+, TP, T-
            [0.4, 1.0, 0.5],  # TP vs T+, TP, T-
            [0.6, 0.9, 1.0]   # T- vs T+, TP, T-
        ], dtype=np.float64),
        'ess_psa': np.array([27272.7272727273], dtype=np.float64),
    }  # patient #1 alpha params (index 7): 0.7,0.8,0.4,0.5,0.6,0.9
    
    # Initialize environment
    env = ProstateCancerTherapyEnv(LV_params)
    observation, info = env.reset(seed=42, options=1)
    ratios_history = []
    psa_norm_history = []
    rewards_history = []
    population_history = []
    cell_counts_history = []
    
    # Store initial values
    ratios_history.append(observation['ratios'].copy())
    psa_norm_history.append(observation['psa'][0])
    population_history.append(observation['population'][0])
    cell_counts_history.append(info['counts'].copy())

    print("Initial observation:")
    print(f"  Ratios: {observation['ratios']}")
    print(f"  Population: {observation['population']}")
    print(f"  PSA: {observation['psa']}")
    print(f"  Carrying Capacities: {env.carrying_capacities}")
    print(f"  Cell counts: {info['counts']}")

    timesteps = 10
    for t in range(timesteps):
        # Take one step with action 0 (no treatment)
        action = 0
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Store data
        ratios_history.append(observation['ratios'].copy())
        psa_norm_history.append(observation['psa'][0])
        rewards_history.append(reward)
        population_history.append(observation['population'][0])
        cell_counts_history.append(info['counts'].copy())

        # Print progress every 100 steps
        if (t + 1) % 100 == 0:
            print(f"Step {t+1}: PSA_norm={observation['psa'][0]:.4f}, Reward={reward:.6f}, Population={observation['population'][0]:.0f}")
        
        if terminated or truncated:
            print(f"Simulation ended at step {t+1}")
            break
    
    # Convert to numpy arrays for easier plotting
    ratios_history = np.array(ratios_history)
    psa_norm_history = np.array(psa_norm_history)
    rewards_history = np.array(rewards_history)
    population_history = np.array(population_history)
    cell_counts_history = np.array(cell_counts_history)
    
    # Create time array
    time_steps = np.arange(len(ratios_history))
    
    # Create the plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Cell Group Dynamics - No Treatment', fontsize=16, fontweight='bold')
    
    # Plot 1: Cell type ratios over time
    ax1 = axes[0, 0]
    ax1.plot(time_steps, ratios_history[:, 0], 'b-', linewidth=2, label='T+ (Androgen dependent)', alpha=0.8)
    ax1.plot(time_steps, ratios_history[:, 1], 'r-', linewidth=2, label='TP (Testosterone producing)', alpha=0.8)
    ax1.plot(time_steps, ratios_history[:, 2], 'g-', linewidth=2, label='T- (Androgen independent)', alpha=0.8)
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Cell Type Ratios')
    ax1.set_title('Cell Type Ratios Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Normalized PSA over time
    ax2 = axes[0, 1]
    ax2.plot(time_steps, psa_norm_history, 'purple', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Normalized PSA')
    ax2.set_title('Normalized PSA Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Initial PSA level')
    ax2.legend()
    
    # Plot 3: Reward over time
    ax3 = axes[1, 0]
    ax3.plot(time_steps[1:], rewards_history, 'orange', linewidth=2, alpha=0.8)
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Reward')
    ax3.set_title('Reward Over Time')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Plot 4: Absolute cell counts over time
    ax4 = axes[1, 1]
    ax4.plot(time_steps, cell_counts_history[:, 0], 'b-', linewidth=2, label='T+ cells', alpha=0.8)
    ax4.plot(time_steps, cell_counts_history[:, 1], 'r-', linewidth=2, label='TP cells', alpha=0.8)
    ax4.plot(time_steps, cell_counts_history[:, 2], 'g-', linewidth=2, label='T- cells', alpha=0.8)
    ax4.set_xlabel('Time Steps')
    ax4.set_ylabel('Cell Count')
    ax4.set_title('Absolute Cell Counts Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')  # Use log scale for better visualization
    
    plt.tight_layout()
    plt.show()
    
    # Print final statistics
    print("\n" + "="*50)
    print("FINAL STATISTICS")
    print("="*50)
    print(f"Initial PSA (normalized): {psa_norm_history[0]:.4f}")
    print(f"Final PSA (normalized): {psa_norm_history[-1]:.4f}")
    print(f"PSA change: {psa_norm_history[-1] - psa_norm_history[0]:.4f}")
    print(f"Total reward: {np.sum(rewards_history):.6f}")
    print(f"Average reward per step: {np.mean(rewards_history):.6f}")
    print(f"Initial population: {population_history[0]:.0f}")
    print(f"Final population: {population_history[-1]:.0f}")
    print(f"Population change: {population_history[-1] - population_history[0]:.0f}")
    
    print("\nFinal cell type ratios:")
    final_ratios = ratios_history[-1]
    print(f"  T+ ratio: {final_ratios[0]:.4f}")
    print(f"  TP ratio: {final_ratios[1]:.4f}")
    print(f"  T- ratio: {final_ratios[2]:.4f}")
    
    print("\nFinal cell counts:")
    final_counts = cell_counts_history[-1]
    print(f"  T+ cells: {final_counts[0]:.0f}")
    print(f"  TP cells: {final_counts[1]:.0f}")
    print(f"  T- cells: {final_counts[2]:.0f}")
    
    # Additional analysis: Check if system reached equilibrium
    if len(psa_norm_history) > 100:
        recent_psa_std = np.std(psa_norm_history[-100:])
        print(f"\nRecent PSA stability (std of last 100 steps): {recent_psa_std:.6f}")
        if recent_psa_std < 0.001:
            print("System appears to have reached equilibrium")
        else:
            print("System is still evolving")

    
if __name__ == "__main__":
    main()