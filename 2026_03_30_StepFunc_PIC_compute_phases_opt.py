# ============================================================
# QSP ANGLE OPTIMIZATION FOR STEP FUNCTION
# ============================================================
# GOAL:
#   Find the QSP angles (theta_opt, phi_opt) that make the
#   quantum circuit output Z = p0-p1 approximate a STEP function.
#
# METHOD:
#   Use scipy L-BFGS-B optimizer to minimize MSE between
#   the circuit output and the arctan surrogate of STEP.
#
# FILE NAMING CONVENTION:
#   All output files include the function name and L value so
#   that angle files for different L never overwrite each other.
#   Example (L=15):
#     theta_step_opt_L15.npy
#     phi_step_opt_L15.npy
#     trig_qsp_step_L15.png
#   The NLFT angle-finder loads these with the matching tag
#   for comparison in its optional Step Pulse 3 panel.
#
# OUTPUT:
#   theta_step_opt_L{polydeg}.npy  -- optimized Ry rotation angles (L+1 values)
#   phi_step_opt_L{polydeg}.npy    -- optimized Rz rotation angles (L+1 values)
#   trig_qsp_step_L{polydeg}.png   -- diagnostic plot
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import perceval as pcvl
import perceval.components as comp

# ============================================================
# Settings
# N_approx : controls sharpness of the arctan surrogate
#            larger = sharper transition at x=0
# polydeg  : number of QSP layers L
#            more layers = better approximation but harder to optimize
#
# FILE_TAG : label embedded in all saved filenames for this run
#            encodes function name and L
#            ensures different L runs never overwrite each other
# ============================================================

N_approx = 100
polydeg  = 15

FILE_TAG = f"step_opt_L{polydeg}"
print(f"Output tag for this run: {FILE_TAG}")

# ============================================================
# Target functions
# ============================================================

def step_surrogate(x):
    # Smooth surrogate for the STEP function (paper Eq. B9).
    # f(x) = (2/pi) * arctan(100*x)
    # Used as optimization target because the true STEP is
    # discontinuous and cannot be fit exactly by a polynomial.
    # Approaches -1 for x<<0 and +1 for x>>0.
    return (2.0 / np.pi) * np.arctan(N_approx * x)

def step_true(x):
    # Ideal STEP function: exactly -1 for x<0, +1 for x>=0.
    # Used only for plotting comparison, not for optimization.
    return np.where(x >= 0, 1.0, -1.0)

# ============================================================
# QSP gate definitions (pure matrix math, no Perceval)
#
# These are the 2x2 unitary matrices used in the QSP circuit.
# They correspond to qubit rotations on the Bloch sphere.
# ============================================================

def Ry_mat(theta):
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s],[s, c]], dtype=complex)

def Rz_mat(phi):
    return np.array([
        [np.exp(-1j*phi/2), 0              ],
        [0,                 np.exp(1j*phi/2)]
    ], dtype=complex)

def A_mat(theta, phi):
    return Ry_mat(theta) @ Rz_mat(phi)

# ============================================================
# QSP circuit simulation (pure matrix math)
#
# Circuit structure (paper Bu et al. 2025, Eq. 1):
#   W(x) = A(t0,p0) * prod_{j=1}^{L} [A(tj,pj) @ Rz(x)]
#
# Input state: |psi_in> = [1, 0]  (photon in mode 0)
# Output:      Z = |psi[0]|^2 - |psi[1]|^2  (measurable)
#
# x is the input signal swept over [-pi, pi].
# The circuit approximates f(x) via the Z measurement.
# ============================================================
# paper_qsp_circuit computes Z for one single x value at a time,
# whatever x_val is passed to it -- it does not sweep x itself.
def paper_qsp_circuit(theta_arr, phi_arr, x_val):
    # Compute QSP circuit output Z = p0-p1 for a given x value.
    # theta_arr : array of L+1 Ry rotation angles
    # phi_arr   : array of L+1 Rz rotation angles
    # x_val     : input signal value (swept during experiment)
    L   = len(theta_arr) - 1
    # Start with initial block A(theta_0, phi_0)
    W   = A_mat(theta_arr[0], phi_arr[0])
    # Apply L layers of [A(theta_j, phi_j) @ Rz(x)]
    for j in range(1, L + 1):
        W = A_mat(theta_arr[j], phi_arr[j]) @ Rz_mat(x_val) @ W
    # Apply circuit to input state |1,0>
    psi = W @ np.array([1.0, 0.0])
    # Return measurable output Z = p0 - p1
    return abs(psi[0])**2 - abs(psi[1])**2

# ============================================================
# Optimization setup
#
# We sample 100 x values in [-pi, pi] and compute the
# surrogate target at each point. The optimizer adjusts
# all 2*(L+1) = 32 angles to minimize MSE between the
# circuit output and the surrogate target.
# ============================================================

# x grid for optimization (100 points is enough and fast)
x_samples   = np.linspace(-np.pi, np.pi, 100)
# Target values: arctan surrogate evaluated at each x
target_vals = np.array([step_surrogate(x) for x in x_samples])

def qsp_loss(params):
    # Loss function for optimization.
    # params: flat array of 2*(L+1) values
    #         first L+1 values  = theta angles
    #         last  L+1 values  = phi angles
    # Returns: MSE between circuit output and surrogate target
    theta_arr = params[:polydeg+1]
    phi_arr   = params[polydeg+1:]
    predicted = np.array([
        paper_qsp_circuit(theta_arr, phi_arr, x)
        for x in x_samples
    ])
    return np.mean((predicted - target_vals)**2)

# ============================================================
# Run optimization
#
# Method: L-BFGS-B (gradient-based, efficient for many params)
# Initial guess: random angles in [-pi, pi]
# seed=42 ensures reproducibility
# maxiter=200 limits computation time
# ============================================================

print("Optimizing QSP angles (1 restart, max 200 iterations)...")
np.random.seed(42)
# Random initial angles for all 2*(L+1) parameters
params0 = np.random.uniform(-np.pi, np.pi, 2*(polydeg+1))

result = minimize(
    qsp_loss,
    params0,
    method='L-BFGS-B',
    options={'maxiter': 200, 'ftol': 1e-10}
)
# Extract optimized angles from result
theta_opt = result.x[:polydeg+1]
phi_opt   = result.x[polydeg+1:]

print(f"Done. MSE = {result.fun:.4e}")
print(f"theta: {theta_opt}")
print(f"phi  : {phi_opt}")

# ============================================================
# Evaluate optimized circuit over full x range [-pi, pi]
# to check quality of the STEP approximation
# ============================================================

x_grid  = np.linspace(-np.pi, np.pi, 300)
f_qsp   = np.array([paper_qsp_circuit(theta_opt, phi_opt, x) for x in x_grid])
f_target = np.array([step_surrogate(x) for x in x_grid])
f_true   = step_true(x_grid)

# MSE vs surrogate (optimization target)
mse = np.mean((f_qsp - f_target)**2)
print(f"MSE Z=p0-p1 vs surrogate: {mse:.4f}")

# ============================================================
# Save optimized angles to disk
#
# Filenames include FILE_TAG (function + L) so that:
#   - Different L values never overwrite each other
#   - STEP / ReLU / SELU angle files stay clearly separated
# The NLFT angle-finder loads these tagged filenames for
# optional comparison in its Step Pulse 3 panel.
# ============================================================
theta_filename = f"theta_{FILE_TAG}.npy"
phi_filename   = f"phi_{FILE_TAG}.npy"

np.save(theta_filename, theta_opt)
np.save(phi_filename,   phi_opt)
print(f"Saved: {theta_filename}")
print(f"Saved: {phi_filename}")

# ============================================================
# Plot results
# Left panel:  circuit output vs surrogate vs true STEP
# Right panel: residual (circuit output - surrogate)
#              should be close to zero if optimization worked
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Trigonometric QSP  STEP L={polydeg}  Z=p0-p1", fontsize=13)

# Left panel: compare all three curves
ax = axes[0]
ax.plot(x_grid, f_true,   'k-',  lw=2,  label="True STEP")
ax.plot(x_grid, f_target, 'g--', lw=2,  label="arctan surrogate")
ax.plot(x_grid, f_qsp,    'b.',  ms=3,
        label=f"Z=p0-p1  MSE={mse:.4f}")
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-1.3, 1.3])
ax.set_xticks([-np.pi, 0, np.pi])
ax.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"], fontsize=12)
ax.set_xlabel(r"$x$", fontsize=12)
ax.set_ylabel("Function value", fontsize=12)
ax.set_title(f"L={polydeg}  Z=p0-p1 measurable", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Right panel: residual between circuit output and surrogate
# A flat line at zero means perfect fit to the surrogate
ax2 = axes[1]
diff = f_qsp - f_target
ax2.plot(x_grid, diff, 'purple', lw=1.5, label=f"residual MSE={mse:.4e}")
ax2.axhline(0, color='k', lw=0.8, linestyle='--')
ax2.fill_between(x_grid, diff, alpha=0.2, color='purple')
ax2.set_xlim([-np.pi, np.pi])
ax2.set_xticks([-np.pi, 0, np.pi])
ax2.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"], fontsize=12)
ax2.set_xlabel(r"$x$", fontsize=12)
ax2.set_ylabel("Residual", fontsize=12)
ax2.set_title("Residual vs surrogate", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()

plot_filename = f"trig_qsp_{FILE_TAG}.png"
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved: {plot_filename}")