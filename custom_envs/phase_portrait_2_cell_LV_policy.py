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
        
        # Policy-related attributes
        self.model = None
        self.policy_grid = None
    
    def test_policy_functionality(self):
        """Test that the policy is working correctly"""
        if self.model is None:
            print("No policy model loaded!")
            return False
            
        print("\n=== POLICY FUNCTIONALITY TEST ===")
        
        # Test action space
        print(f"Action space: {self.model.action_space}")
        print(f"Observation space: {self.model.observation_space}")
        
        # Test several points across the phase space
        test_points = [
            (0, 0),           # Origin
            (1000, 1000),     # Low populations
            (5000, 5000),     # Medium populations  
            (8000, 2000),     # High sensitive, low resistant
            (2000, 8000),     # Low sensitive, high resistant
            (9000, 1000),     # Very high sensitive
            (1000, 9000),     # Very high resistant
            (10000, 0),       # Max sensitive, no resistant
            (0, 10000),       # No sensitive, max resistant
        ]
        
        print("\nTesting policy predictions:")
        actions_seen = set()
        
        for s, r in test_points:
            try:
                obs = self._populations_to_observation(s, r)
                action, confidence = self.predict_policy_action(s, r)
                actions_seen.add(action)
                
                print(f"  S={s:5d}, R={r:5d} -> obs=[{obs[0]:6.3f}, {obs[1]:6.3f}] -> action={action}, conf={confidence:.3f}")
                
            except Exception as e:
                print(f"  S={s:5d}, R={r:5d} -> ERROR: {e}")
        
        print(f"\nActions observed: {sorted(actions_seen)}")
        
        if len(actions_seen) == 1:
            print("WARNING: Policy only produces one action type!")
            print("This might indicate:")
            print("  - Policy is deterministic and always chooses the same action")
            print("  - Issue with observation normalization")
            print("  - Model loading problem")
            return False
        else:
            print("✓ Policy appears to be working correctly")
    def load_policy_model(self, model_path_or_model):
        """
        Load the trained policy model
        
        Parameters:
        - model_path_or_model: Either path to model file or loaded model object
        """
        if isinstance(model_path_or_model, str):
            # Assume it's a path and try to load
            try:
                from stable_baselines3 import PPO
                self.model = PPO.load(model_path_or_model)
                print(f"Successfully loaded PPO model from: {model_path_or_model}")
            except Exception as e:
                print(f"Error loading PPO model: {e}")
                return False
        else:
            # Assume it's already a loaded model
            self.model = model_path_or_model
            print("Policy model loaded successfully")
        
        # Test the policy functionality
        self.test_policy_functionality()
        
        return True
        """
        Load the trained policy model
        
        Parameters:
        - model_path_or_model: Either path to model file or loaded model object
        """
        if isinstance(model_path_or_model, str):
            # Assume it's a path and try to load
            try:
                from stable_baselines3 import PPO
                self.model = PPO.load(model_path_or_model)
                print(f"Successfully loaded PPO model from: {model_path_or_model}")
            except Exception as e:
                print(f"Error loading PPO model: {e}")
                return False
        else:
            # Assume it's already a loaded model
            self.model = model_path_or_model
            print("Policy model loaded successfully")
        
        return True
    
    def _populations_to_observation(self, S, R):
        """
        Convert population counts to normalized observation as used by the policy
        
        Parameters:
        - S, R: Sensitive and Resistant population counts
        
        Returns:
        - obs: Normalized observation array as expected by the policy
        """
        # Based on your code comment, you normalize to [-1, 1]:
        # obs[:-1] = (count_obs / max_count) * 2 - 1
        max_count = self.carrying_capacity
        S_norm = (S / max_count) * 2 - 1
        R_norm = (R / max_count) * 2 - 1
        
        return np.array([S_norm, R_norm], dtype=np.float32)
    
    def predict_policy_action(self, S, R):
        """
        Predict policy action for given population state
        
        Parameters:
        - S, R: Population counts
        
        Returns:
        - action: Policy action (0 = no treatment, 1 = treatment)
        - confidence: Action probability/confidence if available
        """
        if self.model is None:
            raise ValueError("No policy model loaded. Use load_policy_model() first.")
        
        obs = self._populations_to_observation(S, R)
        
        try:
            # Get action and optionally action probabilities
            action, _states = self.model.predict(obs, deterministic=True)
            
            # Handle different action formats (scalar, array, etc.)
            if hasattr(action, 'item'):  # numpy scalar
                action = action.item()
            elif hasattr(action, '__getitem__'):  # array-like
                try:
                    action = action[0]
                except (IndexError, TypeError):
                    pass
            
            action = int(action)
            
            # Try to get action probabilities for confidence measure
            confidence = 1.0  # Default
            try:
                # For PPO, we can get action probabilities
                import torch
                if torch.is_tensor(obs):
                    obs_tensor = obs.unsqueeze(0)
                else:
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)  # Add batch dimension
                
                with torch.no_grad():
                    # Get the policy distribution
                    distribution = self.model.policy.get_distribution(obs_tensor)
                    action_probs = distribution.distribution.probs
                    if action < action_probs.shape[1]:  # Make sure action index is valid
                        confidence = float(action_probs[0][action])
                    
            except Exception as conf_e:
                # If confidence calculation fails, that's okay - use default
                pass
            
            return action, confidence
            
        except Exception as e:
            print(f"Error predicting action for S={S:.1f}, R={R:.1f}: {e}")
            return 0, 0.0
    
    def compute_policy_grid(self, resolution=50):
        """
        Compute policy decisions over a grid of population states
        
        Parameters:
        - resolution: Grid resolution for each dimension
        
        Returns:
        - S_grid, R_grid: Population grids
        - action_grid: Policy actions over the grid
        - confidence_grid: Action confidences over the grid
        """
        if self.model is None:
            raise ValueError("No policy model loaded. Use load_policy_model() first.")
        
        max_S = self.K * 1.2
        max_R = self.K * 1.2
        
        S_range = np.linspace(0, max_S, resolution)
        R_range = np.linspace(0, max_R, resolution)
        S_grid, R_grid = np.meshgrid(S_range, R_range)
        
        action_grid = np.zeros_like(S_grid)
        confidence_grid = np.zeros_like(S_grid)
        
        print(f"Computing policy grid ({resolution}x{resolution})...")
        
        # Debug: Test a few sample points first
        test_points = [(1000, 1000), (5000, 5000), (8000, 2000), (2000, 8000)]
        print("Testing policy on sample points:")
        for s, r in test_points:
            action, conf = self.predict_policy_action(s, r)
            print(f"  S={s}, R={r} -> Action={action}, Confidence={conf:.3f}")
        
        action_counts = {0: 0, 1: 0}
        
        for i in range(S_grid.shape[0]):
            for j in range(S_grid.shape[1]):
                action, confidence = self.predict_policy_action(S_grid[i,j], R_grid[i,j])
                action_grid[i,j] = action
                confidence_grid[i,j] = confidence
                action_counts[action] = action_counts.get(action, 0) + 1
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Completed {i+1}/{resolution} rows")
        
        print(f"Policy grid summary:")
        print(f"  Action 0 (no treatment): {action_counts[0]} points ({100*action_counts[0]/(resolution*resolution):.1f}%)")
        print(f"  Action 1 (treatment): {action_counts[1]} points ({100*action_counts[1]/(resolution*resolution):.1f}%)")
        print(f"  Confidence range: [{confidence_grid.min():.3f}, {confidence_grid.max():.3f}]")
        
        self.policy_grid = {
            'S_grid': S_grid,
            'R_grid': R_grid, 
            'action_grid': action_grid,
            'confidence_grid': confidence_grid
        }
        
        return S_grid, R_grid, action_grid, confidence_grid
    
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
    
    def ode_system_with_policy(self, t, y):
        """
        ODE system where treatment decision is made by the policy
        """
        S, R = np.maximum(y, 1e-12)
        
        if self.model is not None:
            action, _ = self.predict_policy_action(S, R)
            treatment_on = bool(action)
        else:
            treatment_on = False  # Default to no treatment if no policy
        
        return self.ode_system(t, y, treatment_on)
    
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
    
    def create_phase_portrait_with_policy(self, figsize=(20, 6), policy_resolution=50):
        """Create phase portrait with policy overlay"""
        
        if self.model is None:
            print("Warning: No policy model loaded. Creating standard phase portrait.")
            return self.create_phase_portrait(treatment_on=False, figsize=figsize)
        
        # Compute policy grid if not already done
        if self.policy_grid is None:
            self.compute_policy_grid(resolution=policy_resolution)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle('Phase Portrait with Policy Overlay', fontsize=16)
        
        # Set axis limits
        max_S = self.K * 1.2
        max_R = self.K * 1.2
        
        # Get policy grid data
        S_grid = self.policy_grid['S_grid']
        R_grid = self.policy_grid['R_grid']
        action_grid = self.policy_grid['action_grid']
        confidence_grid = self.policy_grid['confidence_grid']
        
        # Plot 1: Policy decisions with confidence
        ax = ax1
        
        # Create policy decision plot - use contourf for better visibility
        treatment_mask = action_grid == 1
        no_treatment_mask = action_grid == 0
        
        print(f"Plotting policy: {np.sum(treatment_mask)} treatment points, {np.sum(no_treatment_mask)} no-treatment points")
        
        # Create filled contour plot for policy regions
        im1 = ax.contourf(S_grid, R_grid, action_grid, levels=[0, 0.5, 1], 
                         colors=['lightblue', 'lightcoral'], alpha=0.7)
        
        # Add contour lines for policy boundaries
        if np.any(treatment_mask) and np.any(no_treatment_mask):
            contour_lines = ax.contour(S_grid, R_grid, action_grid, levels=[0.5], 
                                     colors=['black'], linewidths=2)
            ax.clabel(contour_lines, inline=True, fontsize=10, fmt='Policy Boundary')
        
        # Add manual legend for policy regions
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='lightblue', alpha=0.7, label='No Treatment (Action 0)'),
                          Patch(facecolor='lightcoral', alpha=0.7, label='Treatment (Action 1)')]
        
        # Add policy trajectories
        initial_conditions = [
            [max_S*0.1, max_R*0.1],
            [max_S*0.8, max_R*0.1], 
            [max_S*0.1, max_R*0.8],
            [max_S*0.5, max_R*0.5],
            [max_S*0.3, max_R*0.7],
            [max_S*0.7, max_R*0.3]
        ]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(initial_conditions)))
        
        for i, ic in enumerate(initial_conditions):
            try:
                sol = solve_ivp(
                    self.ode_system_with_policy,
                    (0, 50), ic,
                    method='RK45',
                    rtol=1e-6, atol=1e-8,
                    max_step=0.5
                )
                
                if sol.success and len(sol.y[0]) > 5:
                    ax.plot(sol.y[0], sol.y[1], color=colors[i], linewidth=3, alpha=0.8)
                    ax.scatter(ic[0], ic[1], color=colors[i], s=100, marker='o', 
                              edgecolor='black', linewidth=2)
                    # Add arrow
                    if len(sol.y[0]) > 10:
                        mid_idx = len(sol.y[0]) // 3
                        ax.annotate('', xy=(sol.y[0][mid_idx+5], sol.y[1][mid_idx+5]), 
                                   xytext=(sol.y[0][mid_idx], sol.y[1][mid_idx]),
                                   arrowprops=dict(arrowstyle='->', color=colors[i], lw=3))
            except Exception as e:
                print(f"Error plotting policy trajectory {i}: {e}")
                continue
        
        ax.set_xlabel('Sensitive Population (S)', fontsize=12)
        ax.set_ylabel('Resistant Population (R)', fontsize=12)
        ax.set_title('Policy Decisions & Trajectories', fontsize=13)
        ax.set_xlim(0, max_S)
        ax.set_ylim(0, max_R)
        ax.grid(True, alpha=0.3)
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Plot 2: Policy confidence heatmap
        ax = ax2
        
        im = ax.imshow(confidence_grid, extent=[0, max_S, 0, max_R], 
                      origin='lower', cmap='plasma', alpha=0.8)
        
        # Add contour lines for policy boundaries
        contour = ax.contour(S_grid, R_grid, action_grid, levels=[0.5], 
                           colors='white', linewidths=3)
        ax.clabel(contour, inline=True, fontsize=10, fmt='Policy Boundary')
        
        ax.set_xlabel('Sensitive Population (S)', fontsize=12)
        ax.set_ylabel('Resistant Population (R)', fontsize=12)
        ax.set_title('Policy Confidence Heatmap', fontsize=13)
        ax.set_xlim(0, max_S)
        ax.set_ylim(0, max_R)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Policy Confidence', fontsize=10)
        
        # Plot 3: Combined view with nullclines
        ax = ax3
        
        # Policy background with better visualization
        im3 = ax.contourf(S_grid, R_grid, action_grid, levels=[0, 0.5, 1], 
                         colors=['lightblue', 'lightcoral'], alpha=0.5)
        
        # Add policy boundary
        if np.any(action_grid == 1) and np.any(action_grid == 0):
            boundary = ax.contour(S_grid, R_grid, action_grid, levels=[0.5], 
                                colors=['black'], linewidths=3)
        
        # Compute and plot nullclines
        S_range = np.linspace(0, max_S, 200)
        R_range = np.linspace(0, max_R, 200)
        S_mesh, R_mesh = np.meshgrid(S_range, R_range)
        
        # Compute derivatives for nullclines (assuming off treatment for nullclines)
        DS = np.zeros_like(S_mesh)
        DR = np.zeros_like(R_mesh)
        
        for i in range(S_mesh.shape[0]):
            for j in range(S_mesh.shape[1]):
                derivatives = self.ode_system(0, [S_mesh[i,j], R_mesh[i,j]], False)
                DS[i,j] = derivatives[0]
                DR[i,j] = derivatives[1]
        
        # Plot nullclines
        try:
            cs = ax.contour(S_mesh, R_mesh, DS, levels=[0], colors='darkblue', 
                           linewidths=3, alpha=0.8)
            cr = ax.contour(S_mesh, R_mesh, DR, levels=[0], colors='darkred', 
                           linewidths=3, alpha=0.8)
            ax.plot([], [], 'darkblue', linewidth=3, label='dS/dt = 0')
            ax.plot([], [], 'darkred', linewidth=3, label='dR/dt = 0')
        except:
            print("Could not plot nullclines")
        
        # Find and plot equilibria
        equilibria_off = self.find_equilibria(treatment_on=False)
        for i, eq in enumerate(equilibria_off):
            ax.scatter(eq[0], eq[1], color='black', s=300, marker='*', 
                      edgecolor='yellow', linewidth=2, zorder=10)
        
        ax.set_xlabel('Sensitive Population (S)', fontsize=12)
        ax.set_ylabel('Resistant Population (R)', fontsize=12)
        ax.set_title('Policy Overlay with Nullclines', fontsize=13)
        ax.set_xlim(0, max_S)
        ax.set_ylim(0, max_R)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def create_phase_portrait(self, treatment_on=False, figsize=(16, 6)):
        """Create comprehensive phase portrait (original method)"""
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

# Example usage function with policy integration
def run_policy_analysis_example():
    """Example of how to use the enhanced phase portrait with policy"""
    
    # Create analyzer with same parameters
    growth_rates = [0.035, 0.027]
    death_rates = [0.0, 0.0]
    carrying_capacity = 10000
    drug_efficiency = 1.5
    
    analyzer = LV2PopulationPhasePortrait(
        growth_rates=growth_rates,
        death_rates=death_rates,
        carrying_capacity=carrying_capacity,
        drug_efficiency=drug_efficiency
    )
    
    # Load your trained policy model
    # Replace with your actual model path or loaded model
    # analyzer.load_policy_model("/path/to/your/model.zip")
    # OR if you have a loaded model:
    # analyzer.load_policy_model(your_loaded_model)
    
    print("To use policy overlay, load your model with:")
    print("analyzer.load_policy_model('/path/to/model.zip')")
    print("Then call: analyzer.create_phase_portrait_with_policy()")
    
    return analyzer


# Enhanced example with full integration
def run_complete_policy_analysis(model_path=None, model_object=None):
    """
    Complete example showing how to integrate with your existing workflow
    
    Parameters:
    - model_path: Path to saved PPO model
    - model_object: Already loaded PPO model object
    """
    
    # Create analyzer with your parameters
    growth_rates = [0.035, 0.027]
    death_rates = [0.0, 0.0] 
    carrying_capacity = 10000
    drug_efficiency = 1.5
    
    analyzer = LV2PopulationPhasePortrait(
        growth_rates=growth_rates,
        death_rates=death_rates,
        carrying_capacity=carrying_capacity,
        drug_efficiency=drug_efficiency
    )
    
    # Load policy if provided
    if model_path is not None:
        if analyzer.load_policy_model(model_path):
            print("Creating phase portrait with policy overlay...")
            fig_policy = analyzer.create_phase_portrait_with_policy(
                figsize=(20, 6), 
                policy_resolution=40  # Adjust for speed vs resolution trade-off
            )
            fig_policy.savefig('phase_portrait_with_policy.png', dpi=300, bbox_inches='tight')
            print("Saved: phase_portrait_with_policy.png")
            
            # Also create comparison plots
            fig_off = analyzer.create_phase_portrait(treatment_on=False)
            fig_on = analyzer.create_phase_portrait(treatment_on=True)
            
            fig_off.savefig('phase_portrait_off_treatment.png', dpi=300, bbox_inches='tight')
            fig_on.savefig('phase_portrait_on_treatment.png', dpi=300, bbox_inches='tight')
            
            return analyzer, fig_policy, fig_off, fig_on
    elif model_object is not None:
        if analyzer.load_policy_model(model_object):
            print("Creating phase portrait with policy overlay...")
            fig_policy = analyzer.create_phase_portrait_with_policy(
                figsize=(20, 6),
                policy_resolution=40
            )
            fig_policy.savefig('phase_portrait_with_policy.png', dpi=300, bbox_inches='tight')
            print("Saved: phase_portrait_with_policy.png")
            
            return analyzer, fig_policy
    else:
        print("No model provided. Creating standard phase portraits...")
        fig_off = analyzer.create_phase_portrait(treatment_on=False)
        fig_on = analyzer.create_phase_portrait(treatment_on=True)
        
        fig_off.savefig('phase_portrait_off_treatment.png', dpi=300, bbox_inches='tight')
        fig_on.savefig('phase_portrait_on_treatment.png', dpi=300, bbox_inches='tight')
        
        return analyzer, fig_off, fig_on


# Integration with your existing model loading functions
class PolicyPhasePortraitAnalyzer:
    """
    Wrapper class that integrates with your existing model loading workflow
    """
    
    def __init__(self, base_path):
        self.base_path = base_path
        self.analyzer = None
        
    def find_model_from_config(self, cfg):
        """Find model path based on config information, including support for tuning trials"""
        # Reconstruct experiment folder pattern
        k = cfg.experiment.num_envs
        ac_str = f'{k}ac_' if k > 1 else ''
        run_name = cfg.experiment.name
        env_name = cfg.experiment.env_import.split('-')[0].lower()
        model_type = cfg.evaluation.model_type
        trial_to_eval = cfg.evaluation.trial_to_eval
        
        # Pattern to match experiment folders
        folder_pattern = f"{cfg.algorithm.name.lower()}_{ac_str}{env_name}_{run_name}_*"
        
        # Find matching experiment folders
        matching_folders = list(self.base_path.glob(folder_pattern))
        
        if not matching_folders:
            print(f"No experiment folders found matching pattern: {folder_pattern}")
            return None
        
        # Sort by timestamp (most recent first)
        matching_folders.sort(reverse=True)
        latest_experiment = matching_folders[0]
        
        # Add your logic here to find the specific checkpoint
        # This depends on your folder structure
        return latest_experiment

    def _load_model(self, checkpoint_path):
        """Load PPO model from checkpoint"""
        try:
            from stable_baselines3 import PPO
            model = PPO.load(checkpoint_path)
            print(f"Successfully loaded PPO model from: {checkpoint_path}")
            return model
        except Exception as e:
            print(f"Error loading PPO checkpoint: {e}")
            return None
            
    def create_analyzer_with_policy(self, cfg, growth_rates=[0.035, 0.027], 
                                   death_rates=[0.0, 0.0], carrying_capacity=10000, 
                                   drug_efficiency=1.5):
        """
        Create phase portrait analyzer with loaded policy
        
        Parameters:
        - cfg: Your configuration object
        - growth_rates, death_rates, carrying_capacity, drug_efficiency: Model parameters
        """
        
        # Create the analyzer
        self.analyzer = LV2PopulationPhasePortrait(
            growth_rates=growth_rates,
            death_rates=death_rates,
            carrying_capacity=carrying_capacity,
            drug_efficiency=drug_efficiency
        )
        
        # Find and load the model
        model_path = self.find_model_from_config(cfg)
        if model_path:
            model = self._load_model(model_path)
            if model:
                self.analyzer.load_policy_model(model)
                print("Policy successfully integrated with phase portrait analyzer")
                return True
        
        print("Could not load policy model")
        return False
    
    def generate_analysis(self, policy_resolution=40, save_figures=True):
        """Generate complete analysis with policy overlay"""
        
        if self.analyzer is None:
            print("No analyzer created. Call create_analyzer_with_policy first.")
            return None
            
        if self.analyzer.model is None:
            print("No policy loaded. Creating standard phase portraits...")
            fig_off = self.analyzer.create_phase_portrait(treatment_on=False)
            fig_on = self.analyzer.create_phase_portrait(treatment_on=True)
            
            if save_figures:
                fig_off.savefig('phase_portrait_off_treatment.png', dpi=300, bbox_inches='tight')
                fig_on.savefig('phase_portrait_on_treatment.png', dpi=300, bbox_inches='tight')
                print("Saved standard phase portraits")
            
            return fig_off, fig_on
        else:
            print("Creating phase portrait with policy overlay...")
            fig_policy = self.analyzer.create_phase_portrait_with_policy(
                figsize=(20, 6),
                policy_resolution=policy_resolution
            )
            
            # Also create standard ones for comparison
            fig_off = self.analyzer.create_phase_portrait(treatment_on=False)
            fig_on = self.analyzer.create_phase_portrait(treatment_on=True)
            
            if save_figures:
                fig_policy.savefig('phase_portrait_with_policy.png', dpi=300, bbox_inches='tight')
                fig_off.savefig('phase_portrait_off_treatment.png', dpi=300, bbox_inches='tight') 
                fig_on.savefig('phase_portrait_on_treatment.png', dpi=300, bbox_inches='tight')
                print("Saved all phase portrait figures")
            
            return fig_policy, fig_off, fig_on


# Usage example that fits your workflow:
"""
# Example usage in your existing workflow:

from pathlib import Path

# Initialize the analyzer
base_path = Path("/path/to/your/experiments")
policy_analyzer = PolicyPhasePortraitAnalyzer(base_path)

# Load your config (however you do it)
# cfg = load_your_config()

# Create analyzer with policy
if policy_analyzer.create_analyzer_with_policy(cfg):
    # Generate analysis with policy overlay
    fig_policy, fig_off, fig_on = policy_analyzer.generate_analysis(
        policy_resolution=50,  # Higher resolution = better quality but slower
        save_figures=True
    )
    
    # Show plots
    plt.show()
else:
    print("Failed to load policy, generating standard analysis...")
    # Standard analysis without policy
"""


if __name__ == "__main__":
    # Create analyzer with your parameters
    growth_rates = [0.035, 0.027]
    death_rates = [0.0, 0.0]
    carrying_capacity = 10000
    drug_efficiency = 1.5
    
    analyzer = LV2PopulationPhasePortrait(
        growth_rates=growth_rates,
        death_rates=death_rates,
        carrying_capacity=carrying_capacity,
        drug_efficiency=drug_efficiency
    )
    
    # Load your model - update this path to match your model location
    model_path = "../runs/ppo_lv2populations_lv2pop_1env_2Msteps_ttp_random_init_state_9k_termination_linear_norm_20250921_210935/models/best_model"
    
    if analyzer.load_policy_model(model_path):
        print("Creating phase portrait with policy overlay...")
        
        # Create the policy-enhanced phase portrait
        fig_policy = analyzer.create_phase_portrait_with_policy(
            figsize=(20, 6),
            policy_resolution=50
        )
        
        # Save the figure
        fig_policy.savefig('phase_portrait_with_policy.png', dpi=300, bbox_inches='tight')
        print("Saved: phase_portrait_with_policy.png")
        
        # Also create standard comparisons
        print("Creating standard phase portraits for comparison...")
        fig_off = analyzer.create_phase_portrait(treatment_on=False)
        fig_on = analyzer.create_phase_portrait(treatment_on=True)
        
        fig_off.savefig('phase_portrait_off_treatment.png', dpi=300, bbox_inches='tight')
        fig_on.savefig('phase_portrait_on_treatment.png', dpi=300, bbox_inches='tight')
        
        print("Saved: phase_portrait_off_treatment.png")
        print("Saved: phase_portrait_on_treatment.png")
        
        # Show all plots
        plt.show()
        
    else:
        print("Failed to load policy model. Creating standard phase portraits only...")
        fig_off = analyzer.create_phase_portrait(treatment_on=False)
        fig_on = analyzer.create_phase_portrait(treatment_on=True)
        
        fig_off.savefig('phase_portrait_off_treatment.png', dpi=300, bbox_inches='tight')
        fig_on.savefig('phase_portrait_on_treatment.png', dpi=300, bbox_inches='tight')
        
        print("Saved: phase_portrait_off_treatment.png")
        print("Saved: phase_portrait_on_treatment.png")
        
        plt.show()