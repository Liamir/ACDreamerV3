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

plt.show()