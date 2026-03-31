# ============================================================
# QSP ANGLE OPTIMIZATION -- STEP / ReLU / SELU
# ============================================================
# GOAL:
#   Find the QSP angles (theta_opt, phi_opt) that make the
#   quantum circuit output Z = p0-p1 approximate a target
#   activation function (STEP, ReLU, or SELU).
#
# METHOD:
#   Use scipy L-BFGS-B optimizer to minimize MSE between
#   the circuit output and the smooth surrogate of the target.
#
# HOW TO SWITCH BETWEEN STEP / ReLU / SELU:
#   Change FUNC_NAME at the top of the settings block below.
#   Everything else -- surrogate function, file names, plot
#   titles -- updates automatically.
#   FUNC_NAME = "STEP"   --> approximates STEP function
#   FUNC_NAME = "ReLU"   --> approximates ReLU function
#   FUNC_NAME = "SELU"   --> approximates SELU function
#
# FILE NAMING CONVENTION:
#   All output files include FUNC_NAME and L so that angle files
#   for different functions and different L never overwrite each other.
#   Example (FUNC_NAME="STEP", L=15):
#     theta_step_opt_L15.npy
#     phi_step_opt_L15.npy
#     trig_qsp_step_opt_L15.png
#   The NLFT angle-finder loads these with the matching tag
#   for optional comparison.
#
# OUTPUT:
#   theta_{FUNC_LOWER}_opt_L{polydeg}.npy  -- optimized Ry angles (L+1 values)
#   phi_{FUNC_LOWER}_opt_L{polydeg}.npy    -- optimized Rz angles (L+1 values)
#   trig_qsp_{FUNC_LOWER}_opt_L{polydeg}.png  -- diagnostic plot
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ============================================================
# ---- SETTINGS: change these two lines to switch function ----
#
# FUNC_NAME : "STEP", "ReLU", or "SELU"
#             controls all filenames, titles, and surrogate target
# polydeg   : number of QSP layers L
#             more layers = better approximation but harder to optimize
# ============================================================

FUNC_NAME = "STEP"   # <-- change to "ReLU" or "SELU" as needed
polydeg   = 15

# ============================================================
# Derived settings -- do not change these manually
# FILE_TAG flows into all saved filenames automatically
# ============================================================

FUNC_LOWER = FUNC_NAME.lower()   # "step", "relu", "selu" -- used in filenames
FILE_TAG   = f"{FUNC_LOWER}_opt_L{polydeg}"

print("=" * 62)
print(f"  QSP Angle Optimizer   {FUNC_NAME} Function   L = {polydeg}")
print(f"  Output tag : {FILE_TAG}")
print("=" * 62)

N_approx = 100   # sharpness of arctan surrogate (used for STEP)

# ============================================================
# Target functions
#
# Each FUNC_NAME has its own surrogate (smooth, optimizable)
# and true function (ideal, used only for MSE reporting).
#
# STEP  : surrogate = arctan approximation
# ReLU  : surrogate = softplus approximation
# SELU  : surrogate = scaled softplus approximation
# ============================================================

def get_surrogate(func_name):
    """Return the smooth surrogate function for the given FUNC_NAME."""
    if func_name == "STEP":
        return lambda x: (2.0 / np.pi) * np.arctan(N_approx * x)
    elif func_name == "ReLU":
        # Softplus: smooth approximation of max(0, x)
        return lambda x: np.log(1 + np.exp(N_approx * x)) / N_approx
    elif func_name == "SELU":
        # Scaled ELU surrogate
        alpha = 1.6733
        scale = 1.0507
        return lambda x: scale * np.where(
            x >= 0, x, alpha * (np.exp(x) - 1)
        )
    else:
        raise ValueError(f"Unknown FUNC_NAME: {func_name}. Use 'STEP', 'ReLU', or 'SELU'.")

def get_true_func(func_name):
    """Return the ideal (non-smooth) target function for the given FUNC_NAME."""
    if func_name == "STEP":
        return lambda x: np.where(x >= 0, 1.0, -1.0)
    elif func_name == "ReLU":
        return lambda x: np.maximum(0.0, x)
    elif func_name == "SELU":
        alpha = 1.6733
        scale = 1.0507
        return lambda x: scale * np.where(x >= 0, x, alpha * (np.exp(x) - 1))
    else:
        raise ValueError(f"Unknown FUNC_NAME: {func_name}. Use 'STEP', 'ReLU', or 'SELU'.")

surrogate_func = get_surrogate(FUNC_NAME)
true_func      = get_true_func(FUNC_NAME)

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
# Circuit structure (Bu et al. 2025, Eq. 1):
#   W(x) = A(t0,p0) * prod_{j=1}^{L} [A(tj,pj) @ Rz(x)]
#
# Input state: |psi_in> = [1, 0]  (photon in mode 0)
# Output:      Z = |psi[0]|^2 - |psi[1]|^2  (measurable)
#
# paper_qsp_circuit computes Z for one single x value at a time,
# whatever x_val is passed to it -- it does not sweep x itself.
# ============================================================

def paper_qsp_circuit(theta_arr, phi_arr, x_val):
    # Compute QSP circuit output Z = p0-p1 for a given x value.
    # theta_arr : array of L+1 Ry rotation angles
    # phi_arr   : array of L+1 Rz rotation angles
    # x_val     : input signal value (swept during experiment)
    L = len(theta_arr) - 1
    # Start with initial block A(theta_0, phi_0)
    W = A_mat(theta_arr[0], phi_arr[0])
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
# all 2*(L+1) angles to minimize MSE between the circuit
# output and the surrogate target.
# ============================================================

x_samples   = np.linspace(-np.pi, np.pi, 100)
target_vals = surrogate_func(x_samples)

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

print(f"\nOptimizing QSP angles for {FUNC_NAME} (1 restart, max 200 iterations)...")
np.random.seed(42)
params0 = np.random.uniform(-np.pi, np.pi, 2*(polydeg+1))

result = minimize(
    qsp_loss,
    params0,
    method='L-BFGS-B',
    options={'maxiter': 200, 'ftol': 1e-10}
)

theta_opt = result.x[:polydeg+1]
phi_opt   = result.x[polydeg+1:]

print(f"Done. MSE = {result.fun:.4e}")
print(f"theta: {theta_opt}")
print(f"phi  : {phi_opt}")

# ============================================================
# Evaluate optimized circuit over full x range [-pi, pi]
# ============================================================

x_grid   = np.linspace(-np.pi, np.pi, 300)
f_qsp    = np.array([paper_qsp_circuit(theta_opt, phi_opt, x) for x in x_grid])
f_target = surrogate_func(x_grid)
f_true   = true_func(x_grid)

mse_vs_surrogate = np.mean((f_qsp - f_target)**2)
mse_vs_true      = np.mean((f_qsp - f_true)**2)
print(f"MSE Z=p0-p1 vs surrogate   : {mse_vs_surrogate:.4f}")
print(f"MSE Z=p0-p1 vs true {FUNC_NAME:<5} : {mse_vs_true:.4f}")

# ============================================================
# Save optimized angles to disk
#
# Filenames include FILE_TAG (function + L) so that:
#   - Different L values never overwrite each other
#   - STEP / ReLU / SELU angle files stay clearly separated
# The NLFT angle-finder loads these tagged filenames for
# optional comparison.
# ============================================================

theta_filename = f"theta_{FILE_TAG}.npy"
phi_filename   = f"phi_{FILE_TAG}.npy"

np.save(theta_filename, theta_opt)
np.save(phi_filename,   phi_opt)
print(f"\nSaved: {theta_filename}")
print(f"Saved: {phi_filename}")

# ============================================================
# Plot results
# Left panel:  circuit output vs surrogate vs true function
# Right panel: residual (circuit output - surrogate)
#              should be close to zero if optimization worked
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    f"Trigonometric QSP  {FUNC_NAME}  L={polydeg}  Z=p0-p1",
    fontsize=13
)

xt = [-np.pi, 0, np.pi]
xl = [r"$-\pi$", r"$0$", r"$\pi$"]

# Left panel: compare all three curves
ax = axes[0]
ax.plot(x_grid, f_true,   'k-',  lw=2,
        label=f"True {FUNC_NAME}")
ax.plot(x_grid, f_target, 'g--', lw=2,
        label="surrogate")
ax.plot(x_grid, f_qsp,    'b.',  ms=3,
        label=f"Z=p0-p1  MSE(surrogate)={mse_vs_surrogate:.4f}  "
              f"MSE(true {FUNC_NAME})={mse_vs_true:.4f}")
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-1.3, 1.3])
ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=12)
ax.set_xlabel(r"$x$", fontsize=12)
ax.set_ylabel("Function value", fontsize=12)
ax.set_title(f"{FUNC_NAME}  L={polydeg}  Z=p0-p1 measurable", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Right panel: residual between circuit output and surrogate
# A flat line at zero means perfect fit to the surrogate
ax2 = axes[1]
diff = f_qsp - f_target
ax2.plot(x_grid, diff, 'purple', lw=1.5,
         label=f"residual  MSE={mse_vs_surrogate:.4e}")
ax2.axhline(0, color='k', lw=0.8, linestyle='--')
ax2.fill_between(x_grid, diff, alpha=0.2, color='purple')
ax2.set_xlim([-np.pi, np.pi])
ax2.set_xticks(xt); ax2.set_xticklabels(xl, fontsize=12)
ax2.set_xlabel(r"$x$", fontsize=12)
ax2.set_ylabel("Residual", fontsize=12)
ax2.set_title(f"Residual vs surrogate  ({FUNC_NAME})", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()

plot_filename = f"trig_qsp_{FILE_TAG}.png"
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved: {plot_filename}")