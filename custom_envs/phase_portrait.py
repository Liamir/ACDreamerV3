import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class ProstateCancerPhasePortrait:
    def __init__(self, growth_rates, carrying_capacities, competition_matrix, tp_cap_on_treatment):
        self.growth_rates = growth_rates
        self.carrying_capacities = carrying_capacities
        self.competition_matrix = competition_matrix
        self.tp_cap_on_treatment = tp_cap_on_treatment[0]  # Extract scalar value
        
    def lotka_volterra_3d(self, t, y, treatment_on=False):
        """
        3-cell Lotka-Volterra system: [T+, TP, T-]
        """
        x1, x2, x3 = np.maximum(y, 1e-12)  # Prevent numerical issues
        
        # Adjust carrying capacities based on treatment
        carrying_caps = self.carrying_capacities.copy()
        if treatment_on:
            carrying_caps[0] = max(1.5 * x2, 1.0)  # T+ capacity depends on TP
            carrying_caps[1] = self.tp_cap_on_treatment  # Reduced TP capacity on treatment
        else:
            carrying_caps[0] = max(1.5 * x2, 1.0)  # T+ capacity depends on TP
            # carrying_caps[1] remains at original value (10000)
        
        # Ensure no zero carrying capacities
        carrying_caps = np.maximum(carrying_caps, 1.0)
        
        # Competition effects: Σ(a_ij * x_j) / K_i
        competition_sums = np.dot(self.competition_matrix, [x1, x2, x3])
        competition_effects = competition_sums / carrying_caps
        
        # Lotka-Volterra equations
        dydt = self.growth_rates * [x1, x2, x3] * (1 - competition_effects)
        
        return dydt
    
    def find_nullclines_2d(self, treatment_on=False, resolution=200):
        """
        Find nullclines for the 2D Sensitive vs Resistant system
        Uses a more efficient approach with contour detection
        """
        max_sensitive = 15000
        max_resistant = 12000
        
        # Create grid for contour detection (more efficient than point-by-point)
        sensitive_range = np.linspace(0, max_sensitive, resolution)
        resistant_range = np.linspace(0, max_resistant, resolution)
        S, R = np.meshgrid(sensitive_range, resistant_range)
        
        # Compute derivatives on the entire grid
        DS = np.zeros_like(S)  # Sensitive derivative
        DR = np.zeros_like(R)  # Resistant derivative
        
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                try:
                    derivatives = self.sensitive_resistant_system(0, [S[i,j], R[i,j]], treatment_on)
                    DS[i,j] = derivatives[0]
                    DR[i,j] = derivatives[1]
                except:
                    DS[i,j] = 0
                    DR[i,j] = 0
        
        return S, R, DS, DR  # Return grids for contour plotting
    
    def find_equilibria_2d(self, treatment_on=False):
        """Find equilibria in the 2D Sensitive vs Resistant space"""
        from scipy.optimize import fsolve
        
        def equilibrium_2d(y):
            return self.sensitive_resistant_system(0, y, treatment_on)
        
        # Strategic initial guesses for 2D system
        guesses_2d = [
            [1e-6, 1e-6],     # Near extinction
            [1000, 8000],     # Low sensitive, high resistant
            [8000, 1000],     # High sensitive, low resistant
            [5000, 5000],     # Balanced
            [12000, 2000],    # Very high sensitive
            [2000, 10000],    # Very high resistant
            [100, 100],       # Low both
        ]
        
        equilibria_2d = []
        
        for guess in guesses_2d:
            try:
                eq = fsolve(equilibrium_2d, guess, xtol=1e-10)
                
                # Verify it's actually an equilibrium
                derivatives = equilibrium_2d(eq)
                if np.allclose(derivatives, 0, atol=1e-6) and np.all(eq >= -1e-6):
                    eq = np.maximum(eq, 0)
                    
                    # Check uniqueness
                    is_unique = True
                    for existing_eq in equilibria_2d:
                        if np.allclose(eq, existing_eq, atol=50):  # 50 cell tolerance
                            is_unique = False
                            break
                    
                    if is_unique:
                        equilibria_2d.append(eq)
                        
            except:
                continue
        
        return equilibria_2d  # FIXED: Added return statement
    
    def sensitive_resistant_system(self, t, y, treatment_on=False):
        """
        2D system: Sensitive (T+ + TP) vs Resistant (T-) populations
        y = [Sensitive, Resistant]
        """
        total_sensitive, resistant = np.maximum(y, 1e-12)
        
        # Approximate distribution within sensitive population
        # Use a more realistic split based on growth rates
        total_growth = self.growth_rates[0] + self.growth_rates[1]  # T+ + TP growth rates
        tplus_fraction = self.growth_rates[0] / total_growth
        tp_fraction = self.growth_rates[1] / total_growth
        
        tplus_approx = total_sensitive * tplus_fraction
        tp_approx = total_sensitive * tp_fraction
        
        # Get full 3D dynamics
        full_state = [tplus_approx, tp_approx, resistant]
        full_derivatives = self.lotka_volterra_3d(t, full_state, treatment_on)
        
        # Combine T+ and TP derivatives for sensitive population
        sensitive_derivative = full_derivatives[0] + full_derivatives[1]
        resistant_derivative = full_derivatives[2]
        
        return [sensitive_derivative, resistant_derivative]
    
    def create_phase_portrait_2d(self, treatment_on=False):
        """Create 2D phase portrait: Sensitive vs Resistant with nullclines and equilibria"""
        
        treatment_label = "On Treatment" if treatment_on else "Off Treatment (Holiday)"
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f'Sensitive vs Resistant Population Dynamics - {treatment_label}', fontsize=16)
        
        # Set axis limits
        max_sensitive = 15000
        max_resistant = 12000
        
        # Find nullclines and equilibria
        print(f"Computing nullclines for {treatment_label}...")
        S, R, DS, DR = self.find_nullclines_2d(treatment_on, resolution=150)
        
        print(f"Finding 2D equilibria for {treatment_label}...")
        equilibria_2d = self.find_equilibria_2d(treatment_on)
        print(f"Found {len(equilibria_2d)} 2D equilibria")
        
        # Plot 1: Complete phase portrait with nullclines
        ax = ax1
        
        # Create streamplot first
        x_stream = np.linspace(0, max_sensitive, 20)
        y_stream = np.linspace(0, max_resistant, 15)
        X_stream, Y_stream = np.meshgrid(x_stream, y_stream)
        
        DX_stream = np.zeros_like(X_stream)
        DY_stream = np.zeros_like(Y_stream)
        
        for i in range(X_stream.shape[0]):
            for j in range(X_stream.shape[1]):
                try:
                    derivatives = self.sensitive_resistant_system(0, [X_stream[i,j], Y_stream[i,j]], treatment_on)
                    DX_stream[i,j] = derivatives[0]
                    DY_stream[i,j] = derivatives[1]
                except:
                    DX_stream[i,j] = 0
                    DY_stream[i,j] = 0
        
        # Create streamplot
        ax.streamplot(X_stream, Y_stream, DX_stream, DY_stream, density=1.0, linewidth=1, color='lightgray')
        
        # Plot nullclines using contour
        try:
            # Sensitive nullcline: where dS/dt = 0
            contour_s = ax.contour(S, R, DS, levels=[0], colors='blue', linewidths=3, alpha=0.8)
            ax.plot([], [], 'b-', linewidth=3, label='dS/dt = 0 (Sensitive nullcline)')
            
            # Resistant nullcline: where dR/dt = 0  
            contour_r = ax.contour(S, R, DR, levels=[0], colors='red', linewidths=3, alpha=0.8)
            ax.plot([], [], 'r-', linewidth=3, label='dR/dt = 0 (Resistant nullcline)')
            
            print(f"Successfully plotted nullclines for {treatment_label}")
        except Exception as e:
            print(f"Could not plot nullclines: {e}")
        
        # Plot equilibria with better error handling
        if equilibria_2d and len(equilibria_2d) > 0:
            for i, eq in enumerate(equilibria_2d):
                if eq is not None and len(eq) == 2:
                    ax.scatter(eq[0], eq[1], color='black', s=300, marker='*', 
                              edgecolor='yellow', linewidth=2, zorder=10)
                    
                    # Add text annotation for equilibrium
                    ax.annotate(f'E{i+1}\n({eq[0]:.0f},{eq[1]:.0f})', 
                               xy=(eq[0], eq[1]), xytext=(15, 15), 
                               textcoords='offset points', fontsize=10,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                               zorder=10)
        else:
            print(f"No valid equilibria found for {treatment_label}")
        
        # Add sample trajectories
        # strategic_ics = [
        #     [2000, 1000], [1000, 8000], [8000, 2000], 
        #     [3000, 9000], [6000, 6000], [12000, 1000]
        # ]
        
        # colors = ['darkred', 'darkblue', 'darkgreen', 'darkorange', 'purple', 'brown']
        
        # for i, ic in enumerate(strategic_ics):
        #     try:
        #         sol = solve_ivp(
        #             lambda t, y: self.sensitive_resistant_system(t, y, treatment_on),
        #             (0, 80), ic,
        #             method='RK45',
        #             rtol=1e-6, atol=1e-8,
        #             max_step=1.0
        #         )
                
        #         if sol.success and len(sol.y[0]) > 5:
        #             ax.plot(sol.y[0], sol.y[1], 
        #                    color=colors[i % len(colors)], 
        #                    linewidth=2, alpha=0.7)
        #             ax.scatter(ic[0], ic[1], 
        #                       color=colors[i % len(colors)], 
        #                       s=80, marker='o', edgecolor='black', linewidth=1)
                              
        #     except:
        #         continue
        
        ax.set_xlabel('Sensitive Cells (T+ + TP)', fontsize=12)
        ax.set_ylabel('Resistant Cells (T-)', fontsize=12)
        ax.set_title('Phase Portrait with Nullclines & Equilibria', fontsize=13)
        ax.set_xlim(0, max_sensitive)
        ax.set_ylim(0, max_resistant)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: Vector field with magnitude coloring
        ax = ax2
        
        # Subsample for cleaner vector field
        x_vec = np.linspace(0, max_sensitive, 15)
        y_vec = np.linspace(0, max_resistant, 12)
        X_vec, Y_vec = np.meshgrid(x_vec, y_vec)
        
        DX_vec = np.zeros_like(X_vec)
        DY_vec = np.zeros_like(Y_vec)
        
        for i in range(X_vec.shape[0]):
            for j in range(X_vec.shape[1]):
                point = [X_vec[i,j], Y_vec[i,j]]
                try:
                    derivatives = self.sensitive_resistant_system(0, point, treatment_on)
                    DX_vec[i,j] = derivatives[0]
                    DY_vec[i,j] = derivatives[1]
                except:
                    DX_vec[i,j] = 0
                    DY_vec[i,j] = 0
        
        # Calculate magnitude for color coding
        M = np.sqrt(DX_vec**2 + DY_vec**2)
        M[M == 0] = 1  # Avoid division by zero
        
        # Normalize vectors for consistent arrow sizes
        DX_vec_norm = DX_vec / M
        DY_vec_norm = DY_vec / M
        
        # Plot vector field with magnitude-based coloring
        quiver = ax.quiver(X_vec, Y_vec, DX_vec_norm, DY_vec_norm, M, 
                          scale=25, alpha=0.8, cmap='plasma')
        
        # Add equilibria to vector field plot too
        for i, eq in enumerate(equilibria_2d):
            ax.scatter(eq[0], eq[1], color='white', s=200, marker='*', 
                      edgecolor='black', linewidth=2)
        
        ax.set_xlabel('Sensitive Cells (T+ + TP)', fontsize=12)
        ax.set_ylabel('Resistant Cells (T-)', fontsize=12)
        ax.set_title('Vector Field (colored by speed)', fontsize=13)
        ax.set_xlim(0, max_sensitive)
        ax.set_ylim(0, max_resistant)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for vector magnitude
        cbar = plt.colorbar(quiver, ax=ax, shrink=0.8)
        cbar.set_label('Flow Speed', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def create_3d_phase_portrait(self, treatment_on=False, n_trajectories=12):
        """Create 3D phase portrait for all three cell types"""
        
        treatment_label = "On Treatment" if treatment_on else "Off Treatment (Holiday)"
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Generate trajectories with better initial conditions
        max_val = 6000
        min_val = 200
        np.random.seed(42)
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_trajectories))
        
        # Strategic initial conditions that avoid numerical issues
        strategic_ics = [
            [1000, 2000, 1000],  # Balanced
            [500, 5000, 2000],   # TP-dominated
            [2000, 1000, 4000],  # T- majority
            [3000, 3000, 1000],  # Sensitive majority
            [1000, 1000, 5000],  # Resistant majority
            [4000, 2000, 2000],  # T+ majority
            [2000, 4000, 3000],  # Mixed high
            [1500, 1500, 1500],  # Equal populations
            [500, 1000, 3000],   # Low sensitive
            [4000, 1000, 1000],  # High T+
            [1000, 4000, 1000],  # High TP
            [1000, 1000, 4000],  # High T-
        ]
        
        successful_trajectories = 0
        
        for i, ic in enumerate(strategic_ics):
            try:
                sol = solve_ivp(
                    lambda t, y: self.lotka_volterra_3d(t, y, treatment_on),
                    (0, 50), ic,
                    method='RK45',
                    rtol=1e-5, atol=1e-7,
                    max_step=0.5
                )
                
                if sol.success and len(sol.y[0]) > 10:
                    # Plot trajectory
                    ax.plot(sol.y[0], sol.y[1], sol.y[2], 
                           color=colors[i], linewidth=2, alpha=0.8)
                    
                    # Mark starting point
                    ax.scatter(ic[0], ic[1], ic[2], 
                              color=colors[i], s=100, marker='o', 
                              edgecolor='black', linewidth=1)
                    
                    successful_trajectories += 1
                    
            except Exception as e:
                continue
        
        print(f"3D Portrait: {successful_trajectories} successful trajectories for {treatment_label}")
        
        # Styling
        ax.set_xlabel('T+ Cells', fontsize=12)
        ax.set_ylabel('TP Cells', fontsize=12)
        ax.set_zlabel('T- Cells', fontsize=12)
        ax.set_title(f'3D Phase Portrait - {treatment_label}', fontsize=14)
        
        # Set axis limits
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.set_zlim(0, max_val)
        
        # Better viewing angle
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        return fig
    
    def analyze_equilibria(self, treatment_on=False):
        """Find and analyze equilibrium points"""
        from scipy.optimize import fsolve
        
        def equilibrium_equations(y):
            return self.lotka_volterra_3d(0, y, treatment_on)
        
        # Strategic initial guesses
        initial_guesses = [
            [1e-6, 1e-6, 1e-6],  # Near extinction
            [0, 0, 8000],       # T- dominance
            [0, 5000, 0],       # TP dominance  
            [1000, 1000, 1000], # Balanced
            [100, 5000, 2000],  # TP with some T-
            [2000, 2000, 6000], # Mixed with T- majority
        ]
        
        equilibria = []
        treatment_label = "on treatment" if treatment_on else "off treatment"
        
        print(f"\n=== EQUILIBRIUM ANALYSIS ({treatment_label.upper()}) ===")
        
        for guess in initial_guesses:
            try:
                eq = fsolve(equilibrium_equations, guess, xtol=1e-10)
                derivatives = equilibrium_equations(eq)
                
                if np.allclose(derivatives, 0, atol=1e-6):
                    if np.all(eq >= -1e-6):
                        eq = np.maximum(eq, 0)
                        
                        # Check uniqueness
                        is_unique = True
                        for existing_eq in equilibria:
                            if np.allclose(eq, existing_eq, atol=10):
                                is_unique = False
                                break
                        
                        if is_unique:
                            equilibria.append(eq)
                            total_pop = np.sum(eq)
                            sensitive_pop = eq[0] + eq[1]  # T+ + TP
                            resistant_pop = eq[2]  # T-
                            
                            print(f"Equilibrium {len(equilibria)}:")
                            print(f"  T+ = {eq[0]:.1f}, TP = {eq[1]:.1f}, T- = {eq[2]:.1f}")
                            print(f"  Sensitive = {sensitive_pop:.1f}, Resistant = {resistant_pop:.1f}")
                            
                            # Stability analysis for non-extinction equilibria
                            if total_pop > 1e-3:
                                try:
                                    # Finite difference Jacobian
                                    eps = 1e-6
                                    jacobian = np.zeros((3, 3))
                                    for j in range(3):
                                        eq_plus = eq.copy()
                                        eq_plus[j] += eps
                                        eq_minus = eq.copy() 
                                        eq_minus[j] = max(eq_minus[j] - eps, 1e-12)
                                        
                                        f_plus = equilibrium_equations(eq_plus)
                                        f_minus = equilibrium_equations(eq_minus)
                                        jacobian[:, j] = (f_plus - f_minus) / (2 * eps)
                                    
                                    eigenvals = np.linalg.eigvals(jacobian)
                                    real_parts = np.real(eigenvals)
                                    
                                    if np.all(real_parts < -1e-8):
                                        stability = "Stable"
                                    elif np.all(real_parts > 1e-8):
                                        stability = "Unstable"
                                    else:
                                        stability = "Saddle"
                                    
                                    print(f"  Stability: {stability}")
                                    
                                except:
                                    print(f"  Stability: Analysis failed")
                            else:
                                print(f"  Stability: Extinction state")
                            print()
                            
            except:
                continue
        
        print(f"Found {len(equilibria)} equilibria for {treatment_label}")
        return equilibria
    
    def plot_time_series_comparison(self, initial_condition=[2000, 3000, 1000]):
        """Plot time series comparing treatment vs no treatment"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Treatment vs Holiday Time Series Comparison', fontsize=16)
        
        t_span = (0, 100)
        t_eval = np.linspace(0, 100, 1000)
        
        treatments = [False, True]
        treatment_labels = ['Off Treatment (Holiday)', 'On Treatment']
        
        for treat_idx, (treatment_on, label) in enumerate(zip(treatments, treatment_labels)):
            
            # Solve for individual cell types
            sol = solve_ivp(
                lambda t, y: self.lotka_volterra_3d(t, y, treatment_on),
                t_span, initial_condition,
                t_eval=t_eval,
                method='RK45',
                rtol=1e-6, atol=1e-8
            )
            
            if sol.success:
                # Individual cell types
                ax_individual = axes[treat_idx, 0]
                ax_individual.plot(sol.t, sol.y[0], 'b-', label='T+ (sensitive)', linewidth=2)
                ax_individual.plot(sol.t, sol.y[1], 'g-', label='TP (productive)', linewidth=2) 
                ax_individual.plot(sol.t, sol.y[2], 'r-', label='T- (resistant)', linewidth=2)
                ax_individual.set_xlabel('Time')
                ax_individual.set_ylabel('Cell Count')
                ax_individual.set_title(f'Individual Cell Types - {label}')
                ax_individual.legend()
                ax_individual.grid(True, alpha=0.3)
                
                # Grouped populations  
                ax_grouped = axes[treat_idx, 1]
                sensitive_total = sol.y[0] + sol.y[1]  # T+ + TP
                resistant_total = sol.y[2]  # T-
                
                ax_grouped.plot(sol.t, sensitive_total, 'b-', label='Sensitive (T+ + TP)', linewidth=3)
                ax_grouped.plot(sol.t, resistant_total, 'r-', label='Resistant (T-)', linewidth=3)
                ax_grouped.set_xlabel('Time')
                ax_grouped.set_ylabel('Population Size')
                ax_grouped.set_title(f'Sensitive vs Resistant - {label}')
                ax_grouped.legend()
                ax_grouped.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_all_portraits(self):
        """Generate complete analysis with all visualizations"""
        
        print("PROSTATE CANCER THERAPY PHASE PORTRAIT ANALYSIS")
        print("=" * 60)
        print("Cell Types:")
        print("  T+ (treatment-sensitive)")  
        print("  TP (treatment-productive)")
        print("  T- (treatment-resistant)")
        print()
        print("Parameters:")
        print("  Growth rates (r):", self.growth_rates)
        print("  Competition matrix (A):")
        print(self.competition_matrix)
        print(f"  TP capacity: {self.carrying_capacities[1]} (holiday) → {self.tp_cap_on_treatment} (treatment)")
        print()
        
        # Analyze equilibria for both cases
        eq_off = self.analyze_equilibria(treatment_on=False)
        eq_on = self.analyze_equilibria(treatment_on=True)
        
        print("\nGENERATING VISUALIZATIONS...")
        
        # Create 2D phase portraits (Sensitive vs Resistant)
        fig1 = self.create_phase_portrait_2d(treatment_on=False)  # Off treatment
        fig2 = self.create_phase_portrait_2d(treatment_on=True)   # On treatment
        
        # Create 3D phase portraits  
        fig3 = self.create_3d_phase_portrait(treatment_on=False)  # Off treatment 3D
        fig4 = self.create_3d_phase_portrait(treatment_on=True)   # On treatment 3D
        
        # Create time series comparison
        fig5 = self.plot_time_series_comparison()
        
        # Save all figures with clear names
        fig1.savefig('phase_portrait_2d_off_treatment.png', dpi=300, bbox_inches='tight')
        fig2.savefig('phase_portrait_2d_on_treatment.png', dpi=300, bbox_inches='tight')
        fig3.savefig('phase_portrait_3d_off_treatment.png', dpi=300, bbox_inches='tight')
        fig4.savefig('phase_portrait_3d_on_treatment.png', dpi=300, bbox_inches='tight')
        fig5.savefig('time_series_comparison.png', dpi=300, bbox_inches='tight')
        
        # Summary
        print("\n" + "="*60)
        print("TREATMENT EFFECT SUMMARY")
        print("="*60)
        print(f"Off treatment equilibria: {len(eq_off)}")
        print(f"On treatment equilibria: {len(eq_on)}")
        print()
        print("KEY CLINICAL INSIGHTS:")
        print("1. Treatment reduces TP carrying capacity 100-fold (10,000 → 100)")
        print("2. Off treatment: T- dominance is SADDLE (unstable - recovery possible)")
        print("3. On treatment: T- dominance becomes STABLE (inevitable resistance)")
        print("4. Suggests intermittent therapy may prevent resistance fixation")
        print()
        print("FILES GENERATED:")
        print("- phase_portrait_2d_off_treatment.png")
        print("- phase_portrait_2d_on_treatment.png") 
        print("- phase_portrait_3d_off_treatment.png")
        print("- phase_portrait_3d_on_treatment.png")
        print("- time_series_comparison.png")
        
        return fig1, fig2, fig3, fig4, fig5, eq_off, eq_on



import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

class NullclineDebugger:
    def __init__(self, analyzer):
        """Initialize with the existing analyzer"""
        self.analyzer = analyzer
        
    def debug_nullclines(self, treatment_on=False):
        """
        Debug nullcline calculation with detailed visualization
        """
        treatment_label = "On Treatment" if treatment_on else "Off Treatment"
        
        # Create figure with multiple diagnostic plots
        fig = plt.figure(figsize=(20, 12))
        
        # Define grid
        max_sensitive = 15000
        max_resistant = 12000
        resolution = 50  # Lower resolution for heatmaps
        
        sensitive_range = np.linspace(0, max_sensitive, resolution)
        resistant_range = np.linspace(0, max_resistant, resolution)
        S, R = np.meshgrid(sensitive_range, resistant_range)
        
        # Compute derivatives
        DS = np.zeros_like(S)
        DR = np.zeros_like(R)
        
        print(f"Computing derivatives for {treatment_label}...")
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                derivatives = self.analyzer.sensitive_resistant_system(
                    0, [S[i,j], R[i,j]], treatment_on
                )
                DS[i,j] = derivatives[0]
                DR[i,j] = derivatives[1]
        
        # Plot 1: dS/dt heatmap
        ax1 = plt.subplot(2, 3, 1)
        im1 = ax1.contourf(S, R, DS, levels=20, cmap='RdBu_r', extend='both')
        ax1.contour(S, R, DS, levels=[0], colors='blue', linewidths=3)
        ax1.set_title('dS/dt (Sensitive derivative)')
        ax1.set_xlabel('Sensitive Cells')
        ax1.set_ylabel('Resistant Cells')
        plt.colorbar(im1, ax=ax1)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: dR/dt heatmap
        ax2 = plt.subplot(2, 3, 2)
        im2 = ax2.contourf(S, R, DR, levels=20, cmap='RdBu_r', extend='both')
        ax2.contour(S, R, DR, levels=[0], colors='red', linewidths=3)
        ax2.set_title('dR/dt (Resistant derivative)')
        ax2.set_xlabel('Sensitive Cells')
        ax2.set_ylabel('Resistant Cells')
        plt.colorbar(im2, ax=ax2)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Both nullclines together
        ax3 = plt.subplot(2, 3, 3)
        
        # Try to plot nullclines
        try:
            # Sensitive nullcline
            cs = ax3.contour(S, R, DS, levels=[0], colors='blue', linewidths=2, alpha=0.8)
            ax3.clabel(cs, inline=True, fontsize=8, fmt='dS/dt=0')
        except:
            print("No sensitive nullcline found")
            
        try:
            # Resistant nullcline
            cr = ax3.contour(S, R, DR, levels=[0], colors='red', linewidths=2, alpha=0.8)
            ax3.clabel(cr, inline=True, fontsize=8, fmt='dR/dt=0')
        except:
            print("No resistant nullcline found")
        
        ax3.set_title('Both Nullclines')
        ax3.set_xlabel('Sensitive Cells')
        ax3.set_ylabel('Resistant Cells')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, max_sensitive)
        ax3.set_ylim(0, max_resistant)
        
        # Plot 4: Sign of dS/dt (where is it positive vs negative?)
        ax4 = plt.subplot(2, 3, 4)
        sign_DS = np.sign(DS)
        im4 = ax4.contourf(S, R, sign_DS, levels=[-1.5, -0.5, 0.5, 1.5], 
                           colors=['darkblue', 'lightblue', 'lightcoral', 'darkred'])
        ax4.contour(S, R, DS, levels=[0], colors='black', linewidths=2)
        ax4.set_title('Sign of dS/dt\n(Blue: negative, Red: positive)')
        ax4.set_xlabel('Sensitive Cells')
        ax4.set_ylabel('Resistant Cells')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Sign of dR/dt
        ax5 = plt.subplot(2, 3, 5)
        sign_DR = np.sign(DR)
        im5 = ax5.contourf(S, R, sign_DR, levels=[-1.5, -0.5, 0.5, 1.5], 
                           colors=['darkblue', 'lightblue', 'lightcoral', 'darkred'])
        ax5.contour(S, R, DR, levels=[0], colors='black', linewidths=2)
        ax5.set_title('Sign of dR/dt\n(Blue: negative, Red: positive)')
        ax5.set_xlabel('Sensitive Cells')
        ax5.set_ylabel('Resistant Cells')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Vector field magnitude
        ax6 = plt.subplot(2, 3, 6)
        magnitude = np.sqrt(DS**2 + DR**2)
        im6 = ax6.contourf(S, R, np.log10(magnitude + 1), levels=20, cmap='viridis')
        
        # Add quiver plot
        skip = 3  # Skip some points for clarity
        S_skip = S[::skip, ::skip]
        R_skip = R[::skip, ::skip]
        DS_skip = DS[::skip, ::skip]
        DR_skip = DR[::skip, ::skip]
        
        # Normalize for visibility
        M_skip = np.sqrt(DS_skip**2 + DR_skip**2)
        M_skip[M_skip == 0] = 1
        
        ax6.quiver(S_skip, R_skip, DS_skip/M_skip, DR_skip/M_skip, 
                  M_skip, cmap='plasma', scale=30, alpha=0.6)
        
        ax6.set_title('Vector Field (log magnitude)')
        ax6.set_xlabel('Sensitive Cells')
        ax6.set_ylabel('Resistant Cells')
        plt.colorbar(im6, ax=ax6, label='log10(|derivative|)')
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'Nullcline Diagnostic - {treatment_label}', fontsize=16)
        plt.tight_layout()
        
        # Print diagnostic information
        print(f"\n=== DIAGNOSTIC INFO for {treatment_label} ===")
        print(f"dS/dt range: [{np.min(DS):.2f}, {np.max(DS):.2f}]")
        print(f"dR/dt range: [{np.min(DR):.2f}, {np.max(DR):.2f}]")
        
        # Check if nullclines exist
        has_S_nullcline = np.any(np.abs(DS) < 100)  # Threshold for "near zero"
        has_R_nullcline = np.any(np.abs(DR) < 100)
        
        print(f"Sensitive nullcline exists: {has_S_nullcline}")
        print(f"Resistant nullcline exists: {has_R_nullcline}")
        
        if has_S_nullcline:
            # Find approximate nullcline points
            nullcline_points = np.argwhere(np.abs(DS) < 100)
            if len(nullcline_points) > 0:
                print(f"  Found {len(nullcline_points)} points near dS/dt = 0")
                
        if has_R_nullcline:
            nullcline_points = np.argwhere(np.abs(DR) < 100)
            if len(nullcline_points) > 0:
                print(f"  Found {len(nullcline_points)} points near dR/dt = 0")
        
        return fig, DS, DR, S, R
    
    def test_specific_points(self, treatment_on=False):
        """
        Test derivative values at specific strategic points
        """
        print(f"\n=== TESTING SPECIFIC POINTS ({'On Treatment' if treatment_on else 'Off Treatment'}) ===")
        
        test_points = [
            (0, 0, "Origin"),
            (1000, 1000, "Low both"),
            (5000, 5000, "Medium both"),
            (10000, 1000, "High S, Low R"),
            (1000, 10000, "Low S, High R"),
            (10000, 10000, "High both"),
            (15000, 0, "Max S, Zero R"),
            (0, 12000, "Zero S, Max R"),
        ]
        
        print(f"{'Point':<20} {'S':<8} {'R':<8} {'dS/dt':<12} {'dR/dt':<12}")
        print("-" * 70)
        
        for s, r, label in test_points:
            derivs = self.analyzer.sensitive_resistant_system(0, [s, r], treatment_on)
            print(f"{label:<20} {s:<8} {r:<8} {derivs[0]:<12.2f} {derivs[1]:<12.2f}")
    
    def trace_nullcline_curve(self, treatment_on=False, nullcline_type='sensitive'):
        """
        Trace a nullcline by following where derivative = 0
        """
        print(f"\n=== TRACING {nullcline_type.upper()} NULLCLINE ===")
        
        # Which derivative to track
        deriv_index = 0 if nullcline_type == 'sensitive' else 1
        
        # Try to find nullcline points using root finding
        nullcline_points = []
        
        # Scan along different resistant values
        if nullcline_type == 'sensitive':
            scan_range = np.linspace(0, 12000, 50)
            for r_val in scan_range:
                try:
                    # Find S where dS/dt = 0 for this R
                    def func(s):
                        return self.analyzer.sensitive_resistant_system(0, [s, r_val], treatment_on)[deriv_index]
                    
                    # Try multiple initial guesses
                    for s_guess in [100, 5000, 10000]:
                        try:
                            s_solution = fsolve(func, s_guess, full_output=True)
                            s_val = s_solution[0][0]
                            info = s_solution[1]
                            
                            # Check if solution is valid
                            if info['fvec'][0]**2 < 1e-6 and 0 <= s_val <= 15000:
                                nullcline_points.append([s_val, r_val])
                                break
                        except:
                            continue
                except:
                    continue
        else:
            # Scan along different sensitive values for resistant nullcline
            scan_range = np.linspace(0, 15000, 50)
            for s_val in scan_range:
                try:
                    # Find R where dR/dt = 0 for this S
                    def func(r):
                        return self.analyzer.sensitive_resistant_system(0, [s_val, r], treatment_on)[deriv_index]
                    
                    # Try multiple initial guesses
                    for r_guess in [100, 5000, 10000]:
                        try:
                            r_solution = fsolve(func, r_guess, full_output=True)
                            r_val = r_solution[0][0]
                            info = r_solution[1]
                            
                            # Check if solution is valid
                            if info['fvec'][0]**2 < 1e-6 and 0 <= r_val <= 12000:
                                nullcline_points.append([s_val, r_val])
                                break
                        except:
                            continue
                except:
                    continue
        
        if nullcline_points:
            nullcline_points = np.array(nullcline_points)
            print(f"Found {len(nullcline_points)} points on {nullcline_type} nullcline")
            
            # Plot the traced nullcline
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(nullcline_points[:, 0], nullcline_points[:, 1], 
                      color='red' if nullcline_type == 'resistant' else 'blue',
                      s=20, label=f'{nullcline_type} nullcline points')
            ax.set_xlabel('Sensitive Cells')
            ax.set_ylabel('Resistant Cells')
            ax.set_title(f'Traced {nullcline_type} nullcline - {"On Treatment" if treatment_on else "Off Treatment"}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 15000)
            ax.set_ylim(0, 12000)
            
            return fig, nullcline_points
        else:
            print(f"No {nullcline_type} nullcline found!")
            return None, None




# Initialize system with your parameters
growth_rates = np.array([0.27726, 0.34657, 0.66542], dtype=np.float64)
carrying_capacities = np.array([10000, 10000, 10000], dtype=np.float64)
tp_cap_on_treatment = np.array([100], dtype=np.float64)
competition_matrix = np.array([
    [1.0, 0.7, 0.8],  # T+ vs T+, TP, T-
    [0.4, 1.0, 0.5],  # TP vs T+, TP, T-
    [0.6, 0.9, 1.0]   # T- vs T+, TP, T-
], dtype=np.float64)

# Create analyzer
analyzer = ProstateCancerPhasePortrait(
    growth_rates, carrying_capacities, competition_matrix, tp_cap_on_treatment
)

# Run complete analysis
try:
    figures = analyzer.generate_all_portraits()
    print("\n✓ Analysis completed successfully!")
    # plt.show()  # Added to display the plots
    
except Exception as e:
    print(f"\nError during analysis: {e}")
    import traceback
    traceback.print_exc()

# Create the debugger and run diagnostics
debugger = NullclineDebugger(analyzer)

# Debug both treatment conditions
print("=" * 80)
print("DEBUGGING OFF TREATMENT")
print("=" * 80)
fig_off, DS_off, DR_off, S_off, R_off = debugger.debug_nullclines(treatment_on=False)
debugger.test_specific_points(treatment_on=False)

print("\n" + "=" * 80)
print("DEBUGGING ON TREATMENT")
print("=" * 80)
fig_on, DS_on, DR_on, S_on, R_on = debugger.debug_nullclines(treatment_on=True)
debugger.test_specific_points(treatment_on=True)

# Try to trace nullclines
print("\n" + "=" * 80)
print("TRACING NULLCLINES")
print("=" * 80)

# Trace off-treatment nullclines
fig_trace_s_off, points_s_off = debugger.trace_nullcline_curve(False, 'sensitive')
fig_trace_r_off, points_r_off = debugger.trace_nullcline_curve(False, 'resistant')

# Trace on-treatment nullclines
fig_trace_s_on, points_s_on = debugger.trace_nullcline_curve(True, 'sensitive')
fig_trace_r_on, points_r_on = debugger.trace_nullcline_curve(True, 'resistant')

# plt.show()
# Save all figures instead of showing them
if fig_off:
    fig_off.savefig('debug_nullclines_off_treatment.png', dpi=300, bbox_inches='tight')
    print("Saved: debug_nullclines_off_treatment.png")

if fig_on:
    fig_on.savefig('debug_nullclines_on_treatment.png', dpi=300, bbox_inches='tight')
    print("Saved: debug_nullclines_on_treatment.png")

if fig_trace_s_off:
    fig_trace_s_off.savefig('trace_sensitive_nullcline_off.png', dpi=300, bbox_inches='tight')
    print("Saved: trace_sensitive_nullcline_off.png")

if fig_trace_r_off:
    fig_trace_r_off.savefig('trace_resistant_nullcline_off.png', dpi=300, bbox_inches='tight')
    print("Saved: trace_resistant_nullcline_off.png")

if fig_trace_s_on:
    fig_trace_s_on.savefig('trace_sensitive_nullcline_on.png', dpi=300, bbox_inches='tight')
    print("Saved: trace_sensitive_nullcline_on.png")

if fig_trace_r_on:
    fig_trace_r_on.savefig('trace_resistant_nullcline_on.png', dpi=300, bbox_inches='tight')
    print("Saved: trace_resistant_nullcline_on.png")

# Close all figures to free memory
plt.close('all')