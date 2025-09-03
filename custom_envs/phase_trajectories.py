import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

class LotkaVolterraPhasePortrait:
    def __init__(self, growth_rates, carrying_capacities, competition_matrix, tp_cap_on_treatment):
        self.growth_rates = growth_rates
        self.carrying_capacities = carrying_capacities
        self.competition_matrix = competition_matrix
        self.tp_cap_on_treatment = tp_cap_on_treatment[0]  # Extract scalar value
        
    def lotka_volterra_system(self, t, y, treatment_on=False):
        """
        3-cell Lotka-Volterra system
        y = [T+, TP, T-] cell counts
        """
        x1, x2, x3 = np.maximum(y, 1e-12)  # Prevent negative values and zeros
        
        # Adjust carrying capacities based on treatment
        carrying_caps = self.carrying_capacities.copy()
        if treatment_on:
            # carrying_caps[0] = max(1.5 * x2, 1.0)  # T+ capacity depends on TP, minimum 1
            carrying_caps[0] = 1.5 * x2  # T+ capacity depends on TP, minimum 1
            carrying_caps[1] = self.tp_cap_on_treatment  # Reduced TP capacity
        else:
            # carrying_caps[0] = max(1.5 * x2, 1.0)  # T+ capacity depends on TP, minimum 1
            carrying_caps[0] = 1.5 * x2  # T+ capacity depends on TP, minimum 1
            # carrying_caps[1] remains at original value
        
        # Ensure no zero carrying capacities
        carrying_caps = np.maximum(carrying_caps, 1.0)
        
        # Competition effects: Σ(a_ij * x_j) / K_i
        competition_sums = np.dot(self.competition_matrix, [x1, x2, x3])
        competition_effects = competition_sums / carrying_caps
        
        # Lotka-Volterra equations: dx_i/dt = r_i * x_i * (1 - competition_effect_i)
        dydt = self.growth_rates * [x1, x2, x3] * (1 - competition_effects)
        
        return dydt
    
    def find_equilibria(self, treatment_on=False):
        """Find equilibrium points by solving the system when all derivatives = 0"""
        from scipy.optimize import fsolve
        
        def equilibrium_equations(y):
            return self.lotka_volterra_system(0, y, treatment_on)
        
        # Try multiple initial guesses to find different equilibria
        initial_guesses = [
            [1e-6, 1e-6, 1e-6],  # Near extinction
            [1000, 1000, 1000],  # Mid-range
            [5000, 5000, 5000],  # High
            [100, 5000, 100],  # TP-dominated
            [5000, 100, 5000],  # T+ and T- dominated
            [0, 0, 8000],  # T- only
            [0, 8000, 0],  # TP only
            [8000, 0, 0],  # T+ only (will likely fail due to dependency)
            [2000, 3000, 4000],  # Another mid-range
        ]
        
        equilibria = []
        for guess in initial_guesses:
            try:
                eq = fsolve(equilibrium_equations, guess, xtol=1e-12)
                # Check if it's actually an equilibrium (derivatives near zero)
                derivatives = equilibrium_equations(eq)
                if np.allclose(derivatives, 0, atol=1e-8):
                    # Check if all populations are non-negative
                    if np.all(eq >= -1e-8):  # Allow small numerical errors
                        eq = np.maximum(eq, 0)  # Set small negative values to 0
                        # Check if this equilibrium is already found
                        is_new = True
                        for existing_eq in equilibria:
                            if np.allclose(eq, existing_eq, atol=1e-1):  # More lenient tolerance
                                is_new = False
                                break
                        if is_new:
                            equilibria.append(eq)
            except Exception as e:
                # Skip problematic initial guesses
                continue
        
        return equilibria
    
    def plot_3d_phase_portrait(self, treatment_on=False, n_trajectories=15, t_span=(0, 50)):
        """Generate 3D phase portrait"""
        
        treatment_label = "On Treatment" if treatment_on else "Off Treatment (Holiday)"
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Find equilibria
        equilibria = self.find_equilibria(treatment_on)
        
        # Generate trajectories from random initial conditions
        max_val = 8000
        np.random.seed(42)  # For reproducible results
        
        colors = plt.cm.viridis(np.linspace(0, 1, n_trajectories))
        
        for i in range(n_trajectories):
            # Random initial condition
            init_cond = np.random.uniform(100, max_val, 3)
            
            # Solve ODE
            sol = solve_ivp(
                lambda t, y: self.lotka_volterra_system(t, y, treatment_on),
                t_span, init_cond, 
                dense_output=True, 
                rtol=1e-8, atol=1e-10,
                max_step=0.1
            )
            
            if sol.success:
                # Plot trajectory
                ax.plot(sol.y[0], sol.y[1], sol.y[2], 
                       color=colors[i], alpha=0.7, linewidth=1.5)
                
                # Mark starting point
                ax.scatter(init_cond[0], init_cond[1], init_cond[2], 
                          color=colors[i], s=50, alpha=0.8, marker='o')
        
        # Plot equilibria
        for eq in equilibria:
            ax.scatter(eq[0], eq[1], eq[2], color='red', s=200, 
                      marker='*', edgecolors='black', linewidth=2, 
                      label=f'Equilibrium: ({eq[0]:.0f}, {eq[1]:.0f}, {eq[2]:.0f})')
        
        # Styling
        ax.set_xlabel('T+ Cells', fontsize=12, labelpad=10)
        ax.set_ylabel('TP Cells', fontsize=12, labelpad=10)
        ax.set_zlabel('T- Cells', fontsize=12, labelpad=10)
        ax.set_title(f'3D Phase Portrait - {treatment_label}', fontsize=14, pad=20)
        
        # Set reasonable axis limits
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.set_zlim(0, max_val)
        
        # Add legend if equilibria found
        if equilibria:
            ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
        
        # Improve 3D view angle
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        return fig, ax, equilibria
    
    def plot_2d_projections(self, treatment_on=False, n_trajectories=20, t_span=(0, 50)):
        """Generate 2D projections of the phase portrait"""
        
        treatment_label = "On Treatment" if treatment_on else "Off Treatment (Holiday)"
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Phase Portrait Projections - {treatment_label}', fontsize=16)
        
        # Define projection pairs and labels
        projections = [
            (0, 1, 'T+ Cells', 'TP Cells'),
            (0, 2, 'T+ Cells', 'T- Cells'),
            (1, 2, 'TP Cells', 'T- Cells')
        ]
        
        # Find equilibria
        equilibria = self.find_equilibria(treatment_on)
        
        # Generate trajectories
        max_val = 8000
        np.random.seed(42)
        colors = plt.cm.viridis(np.linspace(0, 1, n_trajectories))
        
        # Plot each 2D projection
        for proj_idx, (i, j, xlabel, ylabel) in enumerate(projections):
            ax = axes[proj_idx // 2, proj_idx % 2]
            
            for traj_idx in range(n_trajectories):
                # Random initial condition
                init_cond = np.random.uniform(100, max_val, 3)
                
                # Solve ODE
                sol = solve_ivp(
                    lambda t, y: self.lotka_volterra_system(t, y, treatment_on),
                    t_span, init_cond, 
                    dense_output=True, 
                    rtol=1e-8, atol=1e-10,
                    max_step=0.1
                )
                
                if sol.success:
                    # Plot 2D projection
                    ax.plot(sol.y[i], sol.y[j], 
                           color=colors[traj_idx], alpha=0.6, linewidth=1.2)
                    
                    # Mark starting point
                    ax.scatter(init_cond[i], init_cond[j], 
                              color=colors[traj_idx], s=30, alpha=0.8)
            
            # Plot equilibria projections
            for eq in equilibria:
                ax.scatter(eq[i], eq[j], color='red', s=150, 
                          marker='*', edgecolors='black', linewidth=1.5)
            
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_xlim(0, max_val)
            ax.set_ylim(0, max_val)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{xlabel} vs {ylabel}', fontsize=12)
        
        # Use the fourth subplot for time series
        ax_time = axes[1, 1]
        
        # Plot a few representative time series
        for traj_idx in range(min(5, n_trajectories)):
            init_cond = np.random.uniform(1000, 6000, 3)
            
            sol = solve_ivp(
                lambda t, y: self.lotka_volterra_system(t, y, treatment_on),
                t_span, init_cond, 
                dense_output=True, 
                rtol=1e-8, atol=1e-10,
                max_step=0.1
            )
            
            if sol.success:
                t_eval = np.linspace(t_span[0], t_span[1], 1000)
                y_eval = sol.sol(t_eval)
                
                ax_time.plot(t_eval, y_eval[0], label='T+' if traj_idx == 0 else '', 
                            color='blue', alpha=0.7, linewidth=1.5)
                ax_time.plot(t_eval, y_eval[1], label='TP' if traj_idx == 0 else '', 
                            color='green', alpha=0.7, linewidth=1.5)
                ax_time.plot(t_eval, y_eval[2], label='T-' if traj_idx == 0 else '', 
                            color='red', alpha=0.7, linewidth=1.5)
        
        ax_time.set_xlabel('Time', fontsize=11)
        ax_time.set_ylabel('Cell Count', fontsize=11)
        ax_time.set_title('Time Series', fontsize=12)
        ax_time.legend()
        ax_time.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, axes, equilibria
    
    def analyze_system(self, treatment_on=False):
        """Analyze the dynamical system properties"""
        
        equilibria = self.find_equilibria(treatment_on)
        treatment_label = "on treatment" if treatment_on else "off treatment"
        
        print(f"\n=== Analysis for {treatment_label.upper()} ===")
        print(f"Growth rates: r = {self.growth_rates}")
        print(f"Competition matrix:")
        print(self.competition_matrix)
        
        if treatment_on:
            print(f"TP carrying capacity (treatment): {self.tp_cap_on_treatment}")
        else:
            print(f"TP carrying capacity (holiday): {self.carrying_capacities[1]}")
        
        print(f"\nFound {len(equilibria)} equilibrium point(s):")
        for i, eq in enumerate(equilibria):
            print(f"  Equilibrium {i+1}: T+ = {eq[0]:.1f}, TP = {eq[1]:.1f}, T- = {eq[2]:.1f}")
            
            # Skip Jacobian analysis for extinction equilibrium to avoid numerical issues
            if np.all(eq < 1e-6):
                print(f"    Stability: Extinction state (analysis skipped)")
                continue
            
            try:
                # Calculate Jacobian at equilibrium for stability analysis
                jacobian = self.compute_jacobian(eq, treatment_on)
                
                if np.all(np.isfinite(jacobian)):
                    eigenvalues = np.linalg.eigvals(jacobian)
                    
                    # Determine stability
                    real_parts = np.real(eigenvalues)
                    if np.all(real_parts < -1e-10):
                        stability = "Stable (sink)"
                    elif np.all(real_parts > 1e-10):
                        stability = "Unstable (source)"
                    elif np.any(real_parts > 1e-10) and np.any(real_parts < -1e-10):
                        stability = "Saddle point"
                    else:
                        stability = "Marginal/Center"
                    
                    print(f"    Eigenvalues: {eigenvalues}")
                    print(f"    Stability: {stability}")
                else:
                    print(f"    Stability: Cannot analyze (numerical issues)")
            except Exception as e:
                print(f"    Stability: Cannot analyze (error: {str(e)[:50]}...)")
        
        return equilibria
    
    def compute_jacobian(self, y, treatment_on=False):
        """Compute Jacobian matrix at point y"""
        x1, x2, x3 = np.maximum(y, 1e-12)  # Prevent zeros
        r1, r2, r3 = self.growth_rates
        
        # Adjust carrying capacities
        carrying_caps = self.carrying_capacities.copy()
        if treatment_on:
            carrying_caps[0] = max(1.5 * x2, 1.0)  # T+ capacity depends on TP, minimum 1
            carrying_caps[1] = self.tp_cap_on_treatment
        else:
            carrying_caps[0] = max(1.5 * x2, 1.0)  # T+ capacity depends on TP, minimum 1
        
        # Ensure no zero carrying capacities
        carrying_caps = np.maximum(carrying_caps, 1.0)
        K1, K2, K3 = carrying_caps
        
        # Competition matrix elements
        a = self.competition_matrix
        
        # Partial derivatives for Lotka-Volterra with variable carrying capacity
        # For f1 = r1*x1*(1 - (a11*x1 + a12*x2 + a13*x3)/K1)
        # where K1 = 1.5*x2, so we need chain rule
        
        competition_term_1 = (a[0,0]*x1 + a[0,1]*x2 + a[0,2]*x3) / K1
        
        # ∂f1/∂x1
        j11 = r1 * (1 - competition_term_1) - r1*x1*(a[0,0]/K1)
        
        # ∂f1/∂x2 (includes dependency of K1 on x2)
        if x2 > 1e-10:  # Only if TP cells exist
            dK1_dx2 = 1.5
            j12 = (-r1*x1*(a[0,1]/K1) + 
                   r1*x1*competition_term_1*(dK1_dx2/K1))
        else:
            j12 = -r1*x1*(a[0,1]/K1)
        
        # ∂f1/∂x3
        j13 = -r1*x1*(a[0,2]/K1)
        
        # For f2 and f3, K2 and K3 are constant
        competition_term_2 = (a[1,0]*x1 + a[1,1]*x2 + a[1,2]*x3) / K2
        competition_term_3 = (a[2,0]*x1 + a[2,1]*x2 + a[2,2]*x3) / K3
        
        j21 = -r2*x2*(a[1,0]/K2)
        j22 = r2 * (1 - competition_term_2) - r2*x2*(a[1,1]/K2)
        j23 = -r2*x2*(a[1,2]/K2)
        
        j31 = -r3*x3*(a[2,0]/K3)
        j32 = -r3*x3*(a[2,1]/K3)
        j33 = r3 * (1 - competition_term_3) - r3*x3*(a[2,2]/K3)
        
        jacobian = np.array([
            [j11, j12, j13],
            [j21, j22, j23],
            [j31, j32, j33]
        ])
        
        # Check for NaN or inf values
        if not np.all(np.isfinite(jacobian)):
            print(f"Warning: Non-finite Jacobian at point {y}")
            jacobian = np.nan_to_num(jacobian, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return jacobian
    
    def generate_phase_portraits(self):
        """Generate both treatment and holiday phase portraits"""
        
        # Analyze both systems
        print("DYNAMICAL SYSTEM ANALYSIS")
        print("=" * 50)
        
        eq_off = self.analyze_system(treatment_on=False)
        eq_on = self.analyze_system(treatment_on=True)
        
        # Generate 3D plots
        fig1, ax1, _ = self.plot_3d_phase_portrait(treatment_on=False)
        fig1.suptitle('3D Phase Portrait - Off Treatment (Holiday)', fontsize=16)
        
        fig2, ax2, _ = self.plot_3d_phase_portrait(treatment_on=True)
        fig2.suptitle('3D Phase Portrait - On Treatment', fontsize=16)
        
        # Generate 2D projections
        fig3, axes3, _ = self.plot_2d_projections(treatment_on=False)
        fig4, axes4, _ = self.plot_2d_projections(treatment_on=True)
        
        # Save figures instead of showing them
        fig1.savefig('phase_portrait_off_treatment_3d.png', dpi=300, bbox_inches='tight')
        fig2.savefig('phase_portrait_on_treatment_3d.png', dpi=300, bbox_inches='tight')  
        fig3.savefig('phase_portrait_off_treatment_2d.png', dpi=300, bbox_inches='tight')
        fig4.savefig('phase_portrait_on_treatment_2d.png', dpi=300, bbox_inches='tight')
        print("Phase portraits saved as PNG files!")
        # plt.show()
        
        return (fig1, fig2, fig3, fig4), (eq_off, eq_on)

# Initialize with your parameters
growth_rates = np.array([0.27726, 0.34657, 0.66542], dtype=np.float64)
carrying_capacities = np.array([10000, 10000, 10000], dtype=np.float64)
tp_cap_on_treatment = np.array([100], dtype=np.float64)
competition_matrix = np.array([
    [1.0, 0.7, 0.8],  # T+ vs T+, TP, T-
    [0.4, 1.0, 0.5],  # TP vs T+, TP, T-
    [0.6, 0.9, 1.0]   # T- vs T+, TP, T-
], dtype=np.float64)

# Create phase portrait generator
phase_portrait = LotkaVolterraPhasePortrait(
    growth_rates, carrying_capacities, competition_matrix, tp_cap_on_treatment
)

# Generate all phase portraits and analysis
figures, equilibria = phase_portrait.generate_phase_portraits()

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Off treatment equilibria: {len(equilibria[0])}")
print(f"On treatment equilibria: {len(equilibria[1])}")
print("\nKey differences between treatment states:")
print("- Off treatment: TP carrying capacity =", carrying_capacities[1])
print("- On treatment: TP carrying capacity =", tp_cap_on_treatment[0])
print("- T+ carrying capacity always depends on TP level (1.5 × TP count)")