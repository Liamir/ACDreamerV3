import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')

class LV2PopulationPhasePortrait:
    def __init__(self, growth_rates, death_rates, carrying_capacity, drug_efficiency):
        """
        Initialize 2-population Lotka-Volterra phase portrait analyzer
        
        Parameters:
        - growth_rates: [r_S, r_R] growth rates for Sensitive and Resistant populations
        - death_rates: [d_S, d_R] death rates for Sensitive and Resistant populations  
        - carrying_capacity: K (shared carrying capacity)
        - drug_efficiency: d_D (drug efficiency, 0-1)
        """
        self.growth_rates = np.array(growth_rates, dtype=np.float64)
        self.death_rates = np.array(death_rates, dtype=np.float64)
        self.carrying_capacity = carrying_capacity
        self.drug_efficiency = drug_efficiency
        
        # Store parameters for easy access
        self.r_S, self.r_R = self.growth_rates
        self.d_S, self.d_R = self.death_rates
        self.K = self.carrying_capacity
        self.d_D = self.drug_efficiency
    
    def ode_system(self, t, y, treatment_on=False):
        """
        2-population Lotka-Volterra ODE system
        
        dS/dt = r_S * S * (1 - (S+R)/K) * (1 - d_D*D) - d_S * S
        dR/dt = r_R * R * (1 - (S+R)/K) - d_R * R
        
        where D = 1 if treatment_on, 0 otherwise
        """
        S, R = np.maximum(y, 1e-12)  # Prevent negative populations
        
        D = 1 if treatment_on else 0
        total_pop = S + R
        
        # Lotka-Volterra equations with logistic growth and treatment effect
        dS_dt = self.r_S * S * (1 - total_pop/self.K) * (1 - self.d_D * D) - self.d_S * S
        dR_dt = self.r_R * R * (1 - total_pop/self.K) - self.d_R * R
        
        return [dS_dt, dR_dt]
    
    def find_equilibria(self, treatment_on=False):
        """Find equilibrium points of the system"""
        def equilibrium_equations(y):
            return self.ode_system(0, y, treatment_on)
        
        # Strategic initial guesses for equilibrium finding
        guesses = [
            [1e-6, 1e-6],           # Near extinction
            [1000, 1000],           # Balanced populations
            [self.K/2, 0],          # Sensitive only
            [0, self.K/2],          # Resistant only
            [self.K/3, self.K/3],   # Mixed populations
            [self.K*0.8, self.K*0.1], # Sensitive dominant
            [self.K*0.1, self.K*0.8], # Resistant dominant
        ]
        
        equilibria = []
        treatment_label = "on treatment" if treatment_on else "off treatment"
        
        print(f"\n=== EQUILIBRIUM ANALYSIS ({treatment_label.upper()}) ===")
        
        for guess in guesses:
            try:
                eq = fsolve(equilibrium_equations, guess, xtol=1e-10)
                derivatives = equilibrium_equations(eq)
                
                # Check if it's actually an equilibrium
                if np.allclose(derivatives, 0, atol=1e-6) and np.all(eq >= -1e-6):
                    eq = np.maximum(eq, 0)  # Ensure non-negative
                    
                    # Check uniqueness
                    is_unique = True
                    for existing_eq in equilibria:
                        if np.allclose(eq, existing_eq, atol=50):  # 50 cell tolerance
                            is_unique = False
                            break
                    
                    if is_unique:
                        equilibria.append(eq)
                        
                        print(f"Equilibrium {len(equilibria)}:")
                        print(f"  S = {eq[0]:.1f}, R = {eq[1]:.1f}")
                        print(f"  Total = {np.sum(eq):.1f}")
                        
                        # Stability analysis
                        self._analyze_stability(eq, treatment_on)
                        print()
                        
            except:
                continue
        
        print(f"Found {len(equilibria)} equilibria for {treatment_label}")
        return equilibria
    
    def _analyze_stability(self, equilibrium, treatment_on):
        """Analyze stability of an equilibrium point"""
        try:
            # Compute Jacobian using finite differences
            eps = 1e-6
            jacobian = np.zeros((2, 2))
            
            for i in range(2):
                eq_plus = equilibrium.copy()
                eq_minus = equilibrium.copy()
                eq_plus[i] += eps
                eq_minus[i] = max(eq_minus[i] - eps, 1e-12)
                
                f_plus = self.ode_system(0, eq_plus, treatment_on)
                f_minus = self.ode_system(0, eq_minus, treatment_on)
                jacobian[:, i] = (np.array(f_plus) - np.array(f_minus)) / (2 * eps)
            
            eigenvals = np.linalg.eigvals(jacobian)
            real_parts = np.real(eigenvals)
            
            if np.all(real_parts < -1e-8):
                stability = "Stable (sink)"
            elif np.all(real_parts > 1e-8):
                stability = "Unstable (source)"
            elif np.any(real_parts > 1e-8) and np.any(real_parts < -1e-8):
                stability = "Saddle point"
            else:
                stability = "Marginal/Center"
            
            print(f"  Stability: {stability}")
            print(f"  Eigenvalues: {eigenvals}")
            
        except:
            print(f"  Stability: Analysis failed")
    
    def create_phase_portrait(self, treatment_on=False, figsize=(16, 6)):
        """Create comprehensive phase portrait"""
        treatment_label = "On Treatment" if treatment_on else "Off Treatment"
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f'2-Population Lotka-Volterra Phase Portrait - {treatment_label}', fontsize=16)
        
        # Set axis limits
        max_S = self.K * 1.2
        max_R = self.K * 1.2
        
        # Find equilibria
        equilibria = self.find_equilibria(treatment_on)
        
        # Plot 1: Phase portrait with nullclines and trajectories
        ax = ax1
        
        # Create meshgrid for nullclines
        S_range = np.linspace(0, max_S, 200)
        R_range = np.linspace(0, max_R, 200)
        S_grid, R_grid = np.meshgrid(S_range, R_range)
        
        # Compute derivatives on grid
        DS = np.zeros_like(S_grid)
        DR = np.zeros_like(R_grid)
        
        for i in range(S_grid.shape[0]):
            for j in range(S_grid.shape[1]):
                derivatives = self.ode_system(0, [S_grid[i,j], R_grid[i,j]], treatment_on)
                DS[i,j] = derivatives[0]
                DR[i,j] = derivatives[1]
        
        # Plot nullclines
        try:
            # S nullcline: where dS/dt = 0
            cs = ax.contour(S_grid, R_grid, DS, levels=[0], colors='blue', linewidths=3, alpha=0.8)
            ax.plot([], [], 'b-', linewidth=3, label='dS/dt = 0 (S nullcline)')
            
            # R nullcline: where dR/dt = 0
            cr = ax.contour(S_grid, R_grid, DR, levels=[0], colors='red', linewidths=3, alpha=0.8)
            ax.plot([], [], 'r-', linewidth=3, label='dR/dt = 0 (R nullcline)')
            
        except:
            print("Could not plot nullclines")
        
        # Plot equilibria
        for i, eq in enumerate(equilibria):
            ax.scatter(eq[0], eq[1], color='black', s=300, marker='*', 
                      edgecolor='yellow', linewidth=2, zorder=10)
            ax.annotate(f'E{i+1}\\n({eq[0]:.0f},{eq[1]:.0f})', 
                       xy=(eq[0], eq[1]), xytext=(15, 15), 
                       textcoords='offset points', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                       zorder=10)
        
        # Add streamplot for flow visualization
        S_stream = np.linspace(0, max_S, 25)
        R_stream = np.linspace(0, max_R, 20)
        S_stream_mesh, R_stream_mesh = np.meshgrid(S_stream, R_stream)
        
        DS_stream = np.zeros_like(S_stream_mesh)
        DR_stream = np.zeros_like(R_stream_mesh)
        
        for i in range(S_stream_mesh.shape[0]):
            for j in range(S_stream_mesh.shape[1]):
                derivatives = self.ode_system(0, [S_stream_mesh[i,j], R_stream_mesh[i,j]], treatment_on)
                DS_stream[i,j] = derivatives[0]
                DR_stream[i,j] = derivatives[1]
        
        # Create streamplot with better visibility
        ax.streamplot(S_stream_mesh, R_stream_mesh, DS_stream, DR_stream, 
                     density=1.2, linewidth=1, color='lightgray',
                     arrowsize=1.5, arrowstyle='->')
        
        # Plot sample trajectories
        initial_conditions = [
            [max_S*0.1, max_R*0.1],
            [max_S*0.8, max_R*0.1], 
            [max_S*0.1, max_R*0.8],
            [max_S*0.5, max_R*0.5],
            [max_S*0.3, max_R*0.7],
            [max_S*0.7, max_R*0.3],
            [max_S*0.9, max_R*0.05],
            [max_S*0.05, max_R*0.9]
        ]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(initial_conditions)))
        
        for i, ic in enumerate(initial_conditions):
            try:
                sol = solve_ivp(
                    lambda t, y: self.ode_system(t, y, treatment_on),
                    (0, 50), ic,
                    method='RK45',
                    rtol=1e-6, atol=1e-8,
                    max_step=0.5
                )
                
                if sol.success and len(sol.y[0]) > 5:
                    ax.plot(sol.y[0], sol.y[1], color=colors[i], linewidth=2, alpha=0.7)
                    ax.scatter(ic[0], ic[1], color=colors[i], s=80, marker='o', 
                              edgecolor='black', linewidth=1)
                    # Add arrow to show direction
                    if len(sol.y[0]) > 10:
                        mid_idx = len(sol.y[0]) // 3
                        ax.annotate('', xy=(sol.y[0][mid_idx+5], sol.y[1][mid_idx+5]), 
                                   xytext=(sol.y[0][mid_idx], sol.y[1][mid_idx]),
                                   arrowprops=dict(arrowstyle='->', color=colors[i], lw=2))
            except:
                continue
        
        ax.set_xlabel('Sensitive Population (S)', fontsize=12)
        ax.set_ylabel('Resistant Population (R)', fontsize=12)
        ax.set_title('Phase Portrait with Nullclines & Trajectories', fontsize=13)
        ax.set_xlim(0, max_S)
        ax.set_ylim(0, max_R)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 2: Vector field
        ax = ax2
        
        # Create coarser grid for vector field
        S_vec = np.linspace(0, max_S, 15)
        R_vec = np.linspace(0, max_R, 12) 
        S_mesh, R_mesh = np.meshgrid(S_vec, R_vec)
        
        DS_vec = np.zeros_like(S_mesh)
        DR_vec = np.zeros_like(R_mesh)
        
        for i in range(S_mesh.shape[0]):
            for j in range(S_mesh.shape[1]):
                derivatives = self.ode_system(0, [S_mesh[i,j], R_mesh[i,j]], treatment_on)
                DS_vec[i,j] = derivatives[0]
                DR_vec[i,j] = derivatives[1]
        
        # Calculate magnitude for color coding
        M = np.sqrt(DS_vec**2 + DR_vec**2)
        M[M == 0] = 1  # Avoid division by zero
        
        # Normalize vectors for consistent arrow sizes
        DS_norm = DS_vec / M
        DR_norm = DR_vec / M
        
        # Plot vector field with magnitude-based coloring
        quiver = ax.quiver(S_mesh, R_mesh, DS_norm, DR_norm, M, 
                          scale=25, alpha=0.8, cmap='plasma')
        
        # Add equilibria to vector field plot
        for i, eq in enumerate(equilibria):
            ax.scatter(eq[0], eq[1], color='white', s=200, marker='*', 
                      edgecolor='black', linewidth=2)
        
        ax.set_xlabel('Sensitive Population (S)', fontsize=12)
        ax.set_ylabel('Resistant Population (R)', fontsize=12)
        ax.set_title('Vector Field (colored by flow speed)', fontsize=13)
        ax.set_xlim(0, max_S)
        ax.set_ylim(0, max_R)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(quiver, ax=ax, shrink=0.8)
        cbar.set_label('Flow Speed', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_time_series_comparison(self, initial_condition=None, t_max=100):
        """Plot time series comparing treatment vs no treatment"""
        
        if initial_condition is None:
            initial_condition = [self.K*0.3, self.K*0.2]  # Default IC
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Treatment vs No Treatment Time Series Comparison', fontsize=16)
        
        t_span = (0, t_max)
        t_eval = np.linspace(0, t_max, 1000)
        
        treatments = [False, True]
        treatment_labels = ['Off Treatment', 'On Treatment']
        axes_pairs = [(ax1, ax2), (ax3, ax4)]
        
        for treat_idx, (treatment_on, label) in enumerate(zip(treatments, treatment_labels)):
            ax_pop, ax_phase = axes_pairs[treat_idx]
            
            # Solve ODE
            sol = solve_ivp(
                lambda t, y: self.ode_system(t, y, treatment_on),
                t_span, initial_condition,
                t_eval=t_eval,
                method='RK45',
                rtol=1e-6, atol=1e-8
            )
            
            if sol.success:
                # Population time series
                ax_pop.plot(sol.t, sol.y[0], 'b-', label='Sensitive', linewidth=2)
                ax_pop.plot(sol.t, sol.y[1], 'r-', label='Resistant', linewidth=2)
                total_pop = sol.y[0] + sol.y[1]
                ax_pop.plot(sol.t, total_pop, 'k--', label='Total', linewidth=2, alpha=0.7)
                
                ax_pop.set_xlabel('Time')
                ax_pop.set_ylabel('Population Size')
                ax_pop.set_title(f'Population Dynamics - {label}')
                ax_pop.legend()
                ax_pop.grid(True, alpha=0.3)
                
                # Phase space trajectory
                ax_phase.plot(sol.y[0], sol.y[1], 'g-', linewidth=2, alpha=0.8)
                ax_phase.scatter(initial_condition[0], initial_condition[1], 
                               color='green', s=100, marker='o', 
                               edgecolor='black', linewidth=2, label='Start')
                ax_phase.scatter(sol.y[0][-1], sol.y[1][-1], 
                               color='red', s=100, marker='s', 
                               edgecolor='black', linewidth=2, label='End')
                
                ax_phase.set_xlabel('Sensitive Population')
                ax_phase.set_ylabel('Resistant Population')
                ax_phase.set_title(f'Phase Space Trajectory - {label}')
                ax_phase.legend()
                ax_phase.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_complete_analysis(self):
        """Generate complete phase portrait analysis"""
        
        print("2-POPULATION LOTKA-VOLTERRA PHASE PORTRAIT ANALYSIS")
        print("=" * 60)
        print("System equations:")
        print(f"dS/dt = r_S * S * (1 - (S+R)/K) * (1 - d_D*D) - d_S * S")
        print(f"dR/dt = r_R * R * (1 - (S+R)/K) - d_R * R")
        print()
        print("Parameters:")
        print(f"  Growth rates: r_S = {self.r_S}, r_R = {self.r_R}")
        print(f"  Death rates: d_S = {self.d_S}, d_R = {self.d_R}")
        print(f"  Carrying capacity: K = {self.K}")
        print(f"  Drug efficiency: d_D = {self.d_D}")
        print()
        
        # Create phase portraits for both treatment conditions
        fig1 = self.create_phase_portrait(treatment_on=False)
        fig2 = self.create_phase_portrait(treatment_on=True)
        
        # Create time series comparison
        fig3 = self.plot_time_series_comparison()
        
        return fig1, fig2, fig3


# Example usage with parameters from your environment
def run_example_analysis():
    """Run example analysis with realistic cancer therapy parameters"""
    
    # Parameters inspired by cancer therapy models
    growth_rates = [0.035, 0.027]      # Sensitive grows faster initially
    death_rates = [0.0, 0.0]      # Both have natural death rates
    carrying_capacity = 10000       # Tissue carrying capacity
    drug_efficiency = 1.5           # Drug reduces sensitive growth by 80%
    
    # Create analyzer
    analyzer = LV2PopulationPhasePortrait(
        growth_rates=growth_rates,
        death_rates=death_rates, 
        carrying_capacity=carrying_capacity,
        drug_efficiency=drug_efficiency
    )
    
    # Generate complete analysis
    fig1, fig2, fig3 = analyzer.generate_complete_analysis()
    
    # Save figures
    fig1.savefig('phase_portrait_off_treatment.png', dpi=300, bbox_inches='tight')
    fig2.savefig('phase_portrait_on_treatment.png', dpi=300, bbox_inches='tight')
    fig3.savefig('time_series_comparison.png', dpi=300, bbox_inches='tight')
    
    print("\nFiles saved:")
    print("- phase_portrait_off_treatment.png")
    print("- phase_portrait_on_treatment.png") 
    print("- time_series_comparison.png")
    
    return analyzer

# Run the example
if __name__ == "__main__":
    analyzer = run_example_analysis()
    plt.show()