"""
Policy-Enabled Phase Portrait Analyzer for Lotka-Volterra System
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.integrate import solve_ivp
from base_analyzer import LV2PopulationBase


class LV2PopulationWithPolicy(LV2PopulationBase):
    """Extended analyzer with policy integration capabilities"""
    
    def __init__(self, growth_rates, death_rates, carrying_capacity, drug_efficiency):
        super().__init__(growth_rates, death_rates, carrying_capacity, drug_efficiency)
        
        # Policy-related attributes
        self.model = None
        self.policy_grid = None
    
    def load_policy_model(self, model_path_or_model):
        """
        Load the trained policy model
        
        Parameters:
        - model_path_or_model: Either path to model file or loaded model object
        """
        if isinstance(model_path_or_model, str):
            try:
                from stable_baselines3 import PPO
                self.model = PPO.load(model_path_or_model)
                print(f"Successfully loaded PPO model from: {model_path_or_model}")
            except Exception as e:
                print(f"Error loading PPO model: {e}")
                return False
        else:
            self.model = model_path_or_model
            print("Policy model loaded successfully")
        
        # Test the policy functionality
        self.test_policy_functionality()
        
        return True
    
    def _populations_to_observation(self, S, R):
        """
        Convert population counts to normalized observation as used by the policy
        
        Parameters:
        - S, R: Sensitive and Resistant population counts
        
        Returns:
        - obs: Normalized observation array as expected by the policy
        """
        # Normalize to [-1, 1] as per your training setup
        max_count = self.K
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
            # Get action from model
            action, _states = self.model.predict(obs, deterministic=True)
            
            # Handle different action formats
            if hasattr(action, 'item'):
                action = action.item()
            elif hasattr(action, '__getitem__'):
                try:
                    action = action[0]
                except (IndexError, TypeError):
                    pass
            
            action = int(action)
            
            # Try to get action probabilities for confidence measure
            confidence = 1.0  # Default confidence
            try:
                import torch
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                
                with torch.no_grad():
                    distribution = self.model.policy.get_distribution(obs_tensor)
                    action_probs = distribution.distribution.probs
                    if action < action_probs.shape[1]:
                        confidence = float(action_probs[0][action])
            except:
                pass  # Use default confidence if calculation fails
            
            return action, confidence
            
        except Exception as e:
            print(f"Error predicting action for S={S:.1f}, R={R:.1f}: {e}")
            return 0, 0.0
    
    def test_policy_functionality(self):
        """Test that the policy is working correctly"""
        if self.model is None:
            print("No policy model loaded!")
            return False
        
        print("\n=== POLICY FUNCTIONALITY TEST ===")
        
        # Test points across the phase space
        test_points = [
            (0, 0),
            (1000, 1000),
            (5000, 5000),
            (8000, 2000),
            (2000, 8000),
            (9000, 1000),
            (1000, 9000),
            (10000, 0),
            (0, 10000),
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
            return True
    
    def compute_policy_grid(self, resolution=100):
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
        
        action_grid = np.zeros_like(S_grid, dtype=int)
        confidence_grid = np.zeros_like(S_grid)
        
        print(f"Computing policy grid ({resolution}x{resolution})...")
        
        action_counts = {0: 0, 1: 0}
        
        for i in range(S_grid.shape[0]):
            for j in range(S_grid.shape[1]):
                action, confidence = self.predict_policy_action(S_grid[i,j], R_grid[i,j])
                action_grid[i,j] = action
                confidence_grid[i,j] = confidence
                action_counts[action] = action_counts.get(action, 0) + 1
                # if action == 0:
                #     print(f'({S_grid[i,j]}, {R_grid[i,j]})')
            
            if (i + 1) % 10 == 0:
                print(f"  Completed {i+1}/{resolution} rows")
        
        print(f"\nPolicy grid summary:")
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
    
    # def ode_system_with_policy(self, t, y):
    #     """ODE system where treatment decision is made by the policy"""
    #     S, R = np.maximum(y, 1e-12)
        
    #     if self.model is not None:
    #         action, _ = self.predict_policy_action(S, R)
    #         treatment_on = bool(action)
    #     else:
    #         treatment_on = False
        
    #     return self.ode_system(t, y, treatment_on)

    def ode_system_with_policy(self, t, y):
        """ODE system where treatment decision is made by the policy"""
        S, R = np.maximum(y, 1e-12)
        
        if self.model is not None:
            # Option 1: Use fresh prediction (current approach)
            action, _ = self.predict_policy_action(S, R)
            
            # Option 2: Use pre-computed grid with interpolation for consistency
            # if self.policy_grid is not None:
            #     action = self._get_action_from_grid(S, R)
            # else:
            #     action, _ = self.predict_policy_action(S, R)
            
            treatment_on = bool(action)
        else:
            treatment_on = False
        
        return self.ode_system(t, y, treatment_on)

    def _get_action_from_grid(self, S, R):
        """Get action from pre-computed grid using nearest neighbor or interpolation"""
        from scipy.interpolate import RegularGridInterpolator
        
        if not hasattr(self, '_policy_interpolator'):
            # Create interpolator once
            S_range = self.policy_grid['S_grid'][0, :]
            R_range = self.policy_grid['R_grid'][:, 0]
            
            # For binary actions, use nearest neighbor
            self._policy_interpolator = RegularGridInterpolator(
                (R_range, S_range), 
                self.policy_grid['action_grid'],
                method='nearest',  # Important: use nearest for binary decisions
                bounds_error=False,
                fill_value=0
            )
        
        point = np.array([R, S])
        action = self._policy_interpolator(point)
        return int(np.round(action))
    
    def create_phase_portrait_with_policy(self, figsize=(20, 6), policy_resolution=100):
        """Create phase portrait with policy overlay - IMPROVED VERSION"""
        
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
        
        # ==== PLOT 1: Policy with Phase Portrait Arrows ====
        ax = ax1
        
        # Show policy regions as background with lighter alpha
        policy_image = ax.imshow(action_grid, extent=[0, max_S, 0, max_R], 
                                origin='lower', cmap='RdBu_r', alpha=0.3, 
                                vmin=0, vmax=1, interpolation='nearest')
        
        # Add straight line constraint: x + y = 9000
        constraint_value = 9000
        x_line = np.linspace(0, min(constraint_value, max_S), 100)
        y_line = constraint_value - x_line

        # Only plot the line where both x and y are within axis limits
        valid_mask = (y_line >= 0) & (y_line <= max_R) & (x_line >= 0) & (x_line <= max_S)
        x_line_valid = x_line[valid_mask]
        y_line_valid = y_line[valid_mask]

        if len(x_line_valid) > 0:
            ax.plot(x_line_valid, y_line_valid, 'k--', linewidth=2, label='$S + R = 9000$')
            ax.legend()

        # Add decision boundary with thicker line
        # if np.any(action_grid == 1) and np.any(action_grid == 0):
        #     contour_lines = ax.contour(S_grid, R_grid, action_grid, levels=[0.5], 
        #                              colors=['black'], linewidths=3, linestyles='--')
        
        # Add streamplot to show dynamics
        S_stream = np.linspace(0, max_S, 50)
        R_stream = np.linspace(0, max_R, 50)
        S_stream_mesh, R_stream_mesh = np.meshgrid(S_stream, R_stream)
        
        DS_stream = np.zeros_like(S_stream_mesh)
        DR_stream = np.zeros_like(R_stream_mesh)
        
        # Compute derivatives WITH policy decisions
        for i in range(S_stream_mesh.shape[0]):
            for j in range(S_stream_mesh.shape[1]):
                derivatives = self.ode_system_with_policy(0, [S_stream_mesh[i,j], R_stream_mesh[i,j]])
                DS_stream[i,j] = derivatives[0]
                DR_stream[i,j] = derivatives[1]
        
        # Create streamplot with better visibility
        stream = ax.streamplot(S_stream_mesh, R_stream_mesh, DS_stream, DR_stream, 
                              density=1.5, linewidth=1.5, color='darkgray',
                              arrowsize=1.8, arrowstyle='->')
        
        # Add sample trajectories with thicker lines
        initial_conditions = [
            # [self.K*0.1, self.K*0.1],
            # [self.K*0.8, self.K*0.01], 
            # [self.K*0.1, self.K*0.5],
            # [self.K*0.6, self.K*0.05],
        ]
        
        colors = ['#2ca02c', '#ff7f0e', '#d62728', '#9467bd']  # Distinct colors
        
        for i, ic in enumerate(initial_conditions):
            try:
                sol = solve_ivp(
                    self.ode_system_with_policy,
                    (0, 300), ic,
                    method='RK45',
                    rtol=1e-6, atol=1e-8,
                    max_step=0.5
                )
                
                if sol.success and len(sol.y[0]) > 5:
                    ax.plot(sol.y[0], sol.y[1], color=colors[i], linewidth=2, 
                           alpha=0.9, zorder=5)
                    ax.scatter(ic[0], ic[1], color=colors[i], s=150, marker='o', 
                              edgecolor='white', linewidth=1, zorder=6)
                    
                    # Add endpoint marker
                    ax.scatter(sol.y[0][-1], sol.y[1][-1], color=colors[i], s=150, 
                              marker='s', edgecolor='white', linewidth=1, zorder=6)
            except Exception as e:
                print(f"Error plotting policy trajectory {i}: {e}")
                continue
        
        # Find and plot equilibria
        equilibria_policy = []
        for eq_guess in [[self.K/2, 0], [0, self.K/2], [self.K/3, self.K/3]]:
            try:
                from scipy.optimize import fsolve
                eq = fsolve(lambda y: self.ode_system_with_policy(0, y), eq_guess, xtol=1e-10)
                if np.all(eq >= 0) and np.sum(eq) <= self.K * 1.2:
                    equilibria_policy.append(eq)
                    ax.scatter(eq[0], eq[1], color='black', s=200, marker='*', 
                              edgecolor='yellow', linewidth=2, zorder=10)
            except:
                pass
        
        # Create legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor='#b40426', alpha=0.3, label='Treatment Region'),
            Patch(facecolor='#3b4cc0', alpha=0.3, label='No Treatment Region'),
            Line2D([0], [0], color='black', linewidth=3, linestyle='--', label='Progression Boundary'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=10, label='Start Points'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
                   markersize=10, label='End Points'),
        ]
        
        ax.set_xlabel('Sensitive Population (S)', fontsize=12)
        ax.set_ylabel('Resistant Population (R)', fontsize=12)
        ax.set_title('Policy Regions with Phase Dynamics', fontsize=13)
        ax.set_xlim(0, max_S)
        ax.set_ylim(0, max_R)
        ax.grid(True, alpha=0.3)
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # ==== PLOT 2: Vector Field with Policy Overlay ====
        ax = ax2
        
        # Show policy regions as very light background
        ax.imshow(action_grid, extent=[0, max_S, 0, max_R], 
                 origin='lower', cmap='RdBu_r', alpha=0.15, 
                 vmin=0, vmax=1, interpolation='nearest')
        
        # Add decision boundary
        if np.any(action_grid == 1) and np.any(action_grid == 0):
            ax.contour(S_grid, R_grid, action_grid, levels=[0.5], 
                      colors=['black'], linewidths=2.5, linestyles='--', alpha=0.7)
        
        # Create vector field grid
        S_vec = np.linspace(max_S*0.05, max_S*0.95, 18)
        R_vec = np.linspace(max_R*0.05, max_R*0.95, 15)
        S_mesh, R_mesh = np.meshgrid(S_vec, R_vec)
        
        DS_vec = np.zeros_like(S_mesh)
        DR_vec = np.zeros_like(R_mesh)
        
        # Compute derivatives WITH policy
        for i in range(S_mesh.shape[0]):
            for j in range(S_mesh.shape[1]):
                derivatives = self.ode_system_with_policy(0, [S_mesh[i,j], R_mesh[i,j]])
                DS_vec[i,j] = derivatives[0]
                DR_vec[i,j] = derivatives[1]
        
        # Calculate magnitude for color and normalization
        M = np.sqrt(DS_vec**2 + DR_vec**2)
        M_safe = np.where(M > 0, M, 1)  # Avoid division by zero
        
        # Create quiver plot with better visibility
        quiver = ax.quiver(S_mesh, R_mesh, DS_vec/M_safe, DR_vec/M_safe, M,
                          scale=20, width=0.004, headwidth=3.5, headlength=4,
                          cmap='viridis', alpha=0.8, edgecolors='black', 
                          linewidths=0.5)
        
        # Add equilibria if any exist
        for eq_guess in [[self.K/2, 0], [0, self.K/2], [self.K/3, self.K/3]]:
            try:
                from scipy.optimize import fsolve
                eq = fsolve(lambda y: self.ode_system_with_policy(0, y), eq_guess, xtol=1e-10)
                if np.all(eq >= 0) and np.sum(eq) <= self.K * 1.2:
                    ax.scatter(eq[0], eq[1], color='red', s=200, marker='*', 
                              edgecolor='white', linewidth=2, zorder=10)
            except:
                pass
        
        ax.set_xlabel('Sensitive Population (S)', fontsize=12)
        ax.set_ylabel('Resistant Population (R)', fontsize=12)
        ax.set_title('Vector Field Under Policy', fontsize=13)
        ax.set_xlim(0, max_S)
        ax.set_ylim(0, max_R)
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(quiver, ax=ax, shrink=0.8)
        cbar.set_label('Flow Speed', fontsize=10)
        
        # ==== PLOT 3: Nullclines with Policy Confidence ====
        ax = ax3
        
        # Show confidence heatmap as background
        im = ax.imshow(confidence_grid, extent=[0, max_S, 0, max_R], 
                      origin='lower', cmap='plasma', alpha=0.6)
        
        # Add policy boundary
        if np.any(action_grid == 1) and np.any(action_grid == 0):
            boundary = ax.contour(S_grid, R_grid, action_grid, levels=[0.5], 
                                colors=['white'], linewidths=3, linestyles='-')
        
        # Compute nullclines for BOTH treatment conditions
        S_range = np.linspace(0, max_S, 200)
        R_range = np.linspace(0, max_R, 200)
        S_mesh, R_mesh = np.meshgrid(S_range, R_range)
        
        # Off-treatment nullclines
        DS_off = np.zeros_like(S_mesh)
        DR_off = np.zeros_like(R_mesh)
        
        # On-treatment nullclines
        DS_on = np.zeros_like(S_mesh)
        DR_on = np.zeros_like(R_mesh)
        
        for i in range(S_mesh.shape[0]):
            for j in range(S_mesh.shape[1]):
                # Off treatment
                deriv_off = self.ode_system(0, [S_mesh[i,j], R_mesh[i,j]], False)
                DS_off[i,j] = deriv_off[0]
                DR_off[i,j] = deriv_off[1]
                
                # On treatment
                deriv_on = self.ode_system(0, [S_mesh[i,j], R_mesh[i,j]], True)
                DS_on[i,j] = deriv_on[0]
                DR_on[i,j] = deriv_on[1]
        
        # Plot nullclines with different styles for on/off treatment
        try:
            # Off-treatment nullclines (solid lines)
            ax.contour(S_mesh, R_mesh, DS_off, levels=[0], colors='darkblue', 
                      linewidths=2.5, linestyles='-', alpha=0.8)
            ax.contour(S_mesh, R_mesh, DR_off, levels=[0], colors='darkred', 
                      linewidths=2.5, linestyles='-', alpha=0.8)
            
            # On-treatment nullclines (dashed lines)
            ax.contour(S_mesh, R_mesh, DS_on, levels=[0], colors='cyan', 
                      linewidths=2.5, linestyles='--', alpha=0.8)
            ax.contour(S_mesh, R_mesh, DR_on, levels=[0], colors='orange', 
                      linewidths=2.5, linestyles='--', alpha=0.8)
            
            # Legend for nullclines
            from matplotlib.lines import Line2D
            nullcline_legend = [
                Line2D([0], [0], color='darkblue', linewidth=2.5, label='dS/dt=0 (off)'),
                Line2D([0], [0], color='darkred', linewidth=2.5, label='dR/dt=0 (off)'),
                Line2D([0], [0], color='cyan', linewidth=2.5, linestyle='--', label='dS/dt=0 (on)'),
                Line2D([0], [0], color='orange', linewidth=2.5, linestyle='--', label='dR/dt=0 (on)'),
                Line2D([0], [0], color='white', linewidth=3, label='Policy boundary'),
            ]
            ax.legend(handles=nullcline_legend, loc='upper right', fontsize=9)
            
        except Exception as e:
            print(f"Could not plot nullclines: {e}")
        
        # Find and plot equilibria under policy
        for eq_guess in [[self.K/2, 0], [0, self.K/2], [self.K/3, self.K/3], [self.K*0.6, self.K*0.3]]:
            try:
                from scipy.optimize import fsolve
                eq = fsolve(lambda y: self.ode_system_with_policy(0, y), eq_guess, xtol=1e-10)
                if np.all(eq >= 0) and np.sum(eq) <= self.K * 1.2:
                    ax.scatter(eq[0], eq[1], color='white', s=200, marker='*', 
                              edgecolor='black', linewidth=2, zorder=10)
            except:
                pass
        
        ax.set_xlabel('Sensitive Population (S)', fontsize=12)
        ax.set_ylabel('Resistant Population (R)', fontsize=12)
        ax.set_title('Nullclines & Policy Confidence', fontsize=13)
        ax.set_xlim(0, max_S)
        ax.set_ylim(0, max_R)
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Policy Confidence', fontsize=10)
        
        plt.tight_layout()
        return fig