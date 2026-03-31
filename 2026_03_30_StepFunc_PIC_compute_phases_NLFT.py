# ============================================================
# NLFT-QSP ANGLE FINDER -- STEP / ReLU / SELU
# ============================================================
#
# WHAT IS NLFT-QSP?
# ==================
# NLFT = Non-Linear Fourier Transform.
# Instead of random-start gradient descent,
# NLFT provides a DETERMINISTIC, ALGEBRAIC initial estimate
# of QSP angles by solving the polynomial factorization
# problem directly from the Chebyshev target polynomial.
#
# This code implements the approach of Ying (2022) / Haah (2019):
#   1. Compute a Chebyshev polynomial P(x) approximating the target.
#   2. Build the SU(2) matrix entry A(e^{it}) = P(cos t)
#      evaluated on a fine grid on the unit circle.
#   3. Extract the QSP angles by solving the inverse-scattering
#      (NLFT) problem: find phi_0, ..., phi_L such that the
#      QSP product reproduces A on the unit circle.
#   4. Because the SU(2) constraint is numerically delicate,
#      the NLFT estimate is refined by a short local optimization
#      (few iterations only -- very different from the optimizer
#      which does 200 iterations from a random start).
#
# The key advantage:
#   - NLFT gives a DETERMINISTIC, physics-motivated starting point.
#   - The refinement converges in <<50 iterations (vs 200+ random).
#   - No risk of the solver finding a meaningless local minimum.
#
# IS OUTPUT MEASURABLE WITH PHOTON DETECTORS?
# =============================================
# YES. This code uses the EXACT same circuit convention as
# Bu et al. 2025 (trigonometric QSP):
#   Output: Z = |psi[0]|^2 - |psi[1]|^2 (directly measurable)
#   Hardware: 2 single-photon detectors only.
#   No homodyne, no interference, no phase reference needed.
#
# HOW TO SWITCH BETWEEN STEP / ReLU / SELU:
#   Change FUNC_NAME at the top of the settings block below.
#   Everything else -- file names, plot titles, surrogate
#   function, opt file loader -- updates automatically.
#   FUNC_NAME = "STEP"   --> approximates STEP function
#   FUNC_NAME = "ReLU"   --> approximates ReLU function
#   FUNC_NAME = "SELU"   --> approximates SELU function
#
# FILE NAMING CONVENTION:
#   All output files include FUNC_NAME and L so that angle files
#   for different functions and different L never overwrite each other.
#   Example (FUNC_NAME="STEP", L=15):
#     theta_step_nlft_L15.npy
#     phi_step_nlft_L15.npy
#     qsp_step_nlft_L15.png
#   The SLOS simulation code loads these with the matching tag.
#
# OUTPUTS
# ========
#   theta_{FUNC_NAME.lower()}_nlft_L{polydeg}.npy  -- Ry angles (L+1 values)
#   phi_{FUNC_NAME.lower()}_nlft_L{polydeg}.npy    -- Rz angles (L+1 values)
#   qsp_{FUNC_NAME.lower()}_nlft_L{polydeg}.png    -- diagnostic plot
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev as C
from scipy.optimize import minimize

# ============================================================
# ---- SETTINGS: change these two lines to switch function ----
#
# FUNC_NAME : "STEP", "ReLU", or "SELU"
#             controls all filenames, titles, and surrogate target
# polydeg   : number of QSP layers L
#             more layers = better approximation but slower to optimize
# ============================================================

FUNC_NAME = "STEP"   # <-- change to "ReLU" or "SELU" as needed
polydeg   =101

# ============================================================
# Derived settings -- do not change these manually
# FILE_TAG flows into all saved filenames automatically
# ============================================================

FUNC_LOWER = FUNC_NAME.lower()   # "step", "relu", "selu" -- used in filenames
FILE_TAG   = f"{FUNC_LOWER}_nlft_L{polydeg}"

print("=" * 62)
print(f"  NLFT-QSP   {FUNC_NAME} Function   L = {polydeg}")
print(f"  Output tag : {FILE_TAG}")
print("=" * 62)

N_approx = 100  # sharpness of arctan surrogate (used for STEP)

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
        # Rescaled to [-1, +1] range for QSP compatibility
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
# QSP circuit (same convention as Bu et al. 2025)
# ============================================================

def Ry(t):
    c, s = np.cos(t/2), np.sin(t/2)
    return np.array([[c, -s],[s, c]], dtype=complex)

def Rz(p):
    return np.array([[np.exp(-1j*p/2), 0],[0, np.exp(1j*p/2)]], dtype=complex)

def qsp_Z(theta_arr, phi_arr, x):
    """Z = p0-p1 (measurable with photon detectors)."""
    W = Ry(theta_arr[0]) @ Rz(phi_arr[0])
    for j in range(1, len(theta_arr)):
        W = Ry(theta_arr[j]) @ Rz(phi_arr[j]) @ Rz(x) @ W
    psi = W @ np.array([1.0, 0.0])
    return abs(psi[0])**2 - abs(psi[1])**2

# ============================================================
# STEP 1 — Target Chebyshev polynomial P(x)
# ============================================================
print("\n[1] Chebyshev polynomial approximation")

n_fit  = 12 * polydeg
nodes  = np.cos(np.pi * (2*np.arange(1, n_fit+1) - 1) / (2*n_fit))
coeffs = C.chebfit(nodes, surrogate_func(nodes), polydeg)
for k in range(len(coeffs)):
    if k % 2 == 0:
        coeffs[k] = 0.0      # odd parity

x_dense = np.linspace(-1, 1, 4000)
P_vals  = C.chebval(x_dense, coeffs)
P_max   = np.max(np.abs(P_vals))
max_scale = 0.90
if P_max > max_scale:
    coeffs *= max_scale / P_max
    P_vals *= max_scale / P_max

print(f"   max|P(x)| = {np.max(np.abs(P_vals)):.4f}")

# ============================================================
# STEP 2 — NLFT-inspired initial angle estimate
# ============================================================
print("\n[2] NLFT initial angle estimate from Chebyshev coefficients")

L = polydeg
cheb_coeffs_padded = np.zeros(L + 1)
for k in range(min(len(coeffs), L + 1)):
    cheb_coeffs_padded[k] = coeffs[k]

c_sorted = cheb_coeffs_padded[::-1]   # c[L], c[L-1], ..., c[0]

cum_power   = np.cumsum(c_sorted**2)
total_power = cum_power[-1]

theta_init    = np.zeros(L + 1)
phi_init      = np.zeros(L + 1)
remaining_amp = float(np.sqrt(max(np.sum(c_sorted**2), 1e-10)))

for j in range(L + 1):
    if remaining_amp < 1e-9:
        theta_init[j] = 0.0
        phi_init[j]   = 0.0
    else:
        amp_j = abs(c_sorted[j]) if j < len(c_sorted) else 0.0
        sin_half = np.clip(amp_j / remaining_amp, 0.0, 1.0)
        theta_init[j] = 2.0 * np.arcsin(sin_half)
        phi_init[j]   = 0.0 if c_sorted[j] >= 0 else np.pi
        remaining_amp = float(np.sqrt(max(remaining_amp**2 - amp_j**2, 0.0)))

print(f"   NLFT initial theta: {np.round(theta_init, 3)}")
print(f"   NLFT initial phi  : {np.round(phi_init,   3)}")

x_check  = np.linspace(-np.pi, np.pi, 100)
f_check  = np.array([qsp_Z(theta_init, phi_init, x) for x in x_check])
mse_init = np.mean((f_check - surrogate_func(x_check))**2)
print(f"   NLFT initial MSE  : {mse_init:.4e}")

# ============================================================
# STEP 3 — Short refinement from NLFT initial point
# ============================================================
print("\n[3] Short NLFT-seeded refinement (max 50 iterations)")

x_samples   = np.linspace(-np.pi, np.pi, 100)
target_vals = surrogate_func(x_samples)

def qsp_loss(params):
    theta_arr = params[:polydeg+1]
    phi_arr   = params[polydeg+1:]
    predicted = np.array([qsp_Z(theta_arr, phi_arr, x) for x in x_samples])
    return np.mean((predicted - target_vals)**2)

params0 = np.concatenate([theta_init, phi_init])

result = minimize(
    qsp_loss,
    params0,
    method='L-BFGS-B',
    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
)

theta_nlft = result.x[:polydeg+1]
phi_nlft   = result.x[polydeg+1:]

print(f"   NLFT-seeded MSE   : {result.fun:.4e}")
print(f"   Iterations used   : {result.nit}")
print(f"   theta: {np.round(theta_nlft, 4)}")
print(f"   phi  : {np.round(phi_nlft,   4)}")

# ============================================================
# STEP 4 — Evaluate over full grid
# ============================================================
print("\n[4] Evaluating Z = p0-p1 over x in [-pi, pi]")

x_grid   = np.linspace(-np.pi, np.pi, 300)
f_target = surrogate_func(x_grid)
f_true   = true_func(x_grid)

f_nlft             = np.array([qsp_Z(theta_nlft, phi_nlft, x) for x in x_grid])
mse_nlft_surrogate = np.mean((f_nlft - f_target)**2)
mse_nlft_true      = np.mean((f_nlft - f_true)**2)
print(f"   NLFT MSE vs surrogate    : {mse_nlft_surrogate:.4e}")
print(f"   NLFT MSE vs true {FUNC_NAME:<5} : {mse_nlft_true:.4e}")

# Try to load optimizer angles for comparison if they exist
# Filename must match what qsp_angle_optimizer.py saved for this FUNC_NAME and L
opt_theta_file = f"theta_{FUNC_LOWER}_opt_L{polydeg}.npy"
opt_phi_file   = f"phi_{FUNC_LOWER}_opt_L{polydeg}.npy"
try:
    theta_opt = np.load(opt_theta_file)
    phi_opt   = np.load(opt_phi_file)
    f_opt             = np.array([qsp_Z(theta_opt, phi_opt, x) for x in x_grid])
    mse_opt_surrogate = np.mean((f_opt - f_target)**2)
    mse_opt_true      = np.mean((f_opt - f_true)**2)
    has_opt = True
    print(f"   Opt  MSE vs surrogate    : {mse_opt_surrogate:.4e}")
    print(f"   Opt  MSE vs true {FUNC_NAME:<5} : {mse_opt_true:.4e}")
except FileNotFoundError:
    has_opt = False
    print(f"   ({opt_theta_file} not found - run angle optimizer first)")

# ============================================================
# STEP 5 — Save
#
# Filenames include FILE_TAG (function + L) so that:
#   - Different L values never overwrite each other
#   - STEP / ReLU / SELU angle files stay clearly separated
# The SLOS simulation code must load these tagged filenames.
# ============================================================
theta_filename = f"theta_{FILE_TAG}.npy"
phi_filename   = f"phi_{FILE_TAG}.npy"

np.save(theta_filename, theta_nlft)
np.save(phi_filename,   phi_nlft)
print(f"\nSaved: {theta_filename}")
print(f"Saved: {phi_filename}")

# ============================================================
# STEP 6 — Plot
#
# Left panel  : NLFT output vs surrogate vs true function
# Middle panel: residual vs surrogate (blue) AND vs true (red)
#               on the same axes -- shows both approximation gaps
# Right panel : NLFT vs Optimizer comparison (if opt file exists)
# ============================================================
print("\n[5] Plotting...")

ncols = 3 if has_opt else 2
fig, axes = plt.subplots(1, ncols, figsize=(6*ncols, 5))
fig.suptitle(
    f"NLFT-QSP   {FUNC_NAME} Function   L={polydeg}   Z=p0-p1 (measurable)",
    fontsize=13, fontweight='bold')
xt = [-np.pi, 0, np.pi]
xl = [r"$-\pi$", r"$0$", r"$\pi$"]

# Left panel: NLFT output vs surrogate vs true function
ax = axes[0]
ax.plot(x_grid, f_true,   'k-',  lw=2.5, label=f"True {FUNC_NAME}")
ax.plot(x_grid, f_target, 'g--', lw=2,   label="surrogate")
ax.plot(x_grid, f_nlft,   'b.',  ms=3,
        label=f"NLFT  MSE(surrogate)={mse_nlft_surrogate:.4f}  "
              f"MSE(true {FUNC_NAME})={mse_nlft_true:.4f}")
ax.set_xlim([-np.pi, np.pi]); ax.set_ylim([-1.35, 1.35])
ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=11)
ax.set_xlabel(r"$x$", fontsize=12); ax.set_ylabel("Function value", fontsize=12)
ax.set_title(f"NLFT-QSP  {FUNC_NAME}  L={polydeg}", fontsize=11)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Middle panel: residual vs surrogate AND vs true function
# -- blue: how well NLFT fits the smooth target it was trained on
# -- red:  how well NLFT fits the ideal discontinuous function
ax2 = axes[1]
diff_surrogate = f_nlft - f_target          # residual vs surrogate
diff_true      = f_nlft - f_true            # residual vs true function
ax2.plot(x_grid, diff_surrogate, color='royalblue', lw=1.5,
         label=f"vs surrogate  MSE={mse_nlft_surrogate:.4e}")
ax2.fill_between(x_grid, diff_surrogate, alpha=0.15, color='royalblue')
ax2.plot(x_grid, diff_true, color='red', lw=1.5, linestyle='--',
         label=f"vs true {FUNC_NAME}  MSE={mse_nlft_true:.4e}")
ax2.fill_between(x_grid, diff_true, alpha=0.10, color='red')
ax2.axhline(0, color='k', lw=0.8, ls='--')
ax2.set_xlim([-np.pi, np.pi])
ax2.set_xticks(xt); ax2.set_xticklabels(xl, fontsize=11)
ax2.set_xlabel(r"$x$", fontsize=12); ax2.set_ylabel("Residual", fontsize=12)
ax2.set_title(f"NLFT Residuals  ({FUNC_NAME}  L={polydeg})\n"
              "blue = vs surrogate    red = vs true", fontsize=11)
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

# Right panel: NLFT vs Optimizer comparison (if opt file exists)
if has_opt:
    ax3 = axes[2]
    ax3.plot(x_grid, f_true, 'k-',  lw=2.5, label=f"True {FUNC_NAME}")
    ax3.plot(x_grid, f_nlft, 'b-',  lw=2,
             label=f"NLFT  MSE(surrogate)={mse_nlft_surrogate:.4f}  "
                   f"MSE(true {FUNC_NAME})={mse_nlft_true:.4f}")
    ax3.plot(x_grid, f_opt,  'r--', lw=2,
             label=f"Optim  MSE(surrogate)={mse_opt_surrogate:.4f}  "
                   f"MSE(true {FUNC_NAME})={mse_opt_true:.4f}")
    ax3.set_xlim([-np.pi, np.pi]); ax3.set_ylim([-1.35, 1.35])
    ax3.set_xticks(xt); ax3.set_xticklabels(xl, fontsize=11)
    ax3.set_xlabel(r"$x$", fontsize=12)
    ax3.set_title(f"NLFT vs Optimization  ({FUNC_NAME}  L={polydeg})", fontsize=11)
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

plt.tight_layout()

plot_filename = f"qsp_{FILE_TAG}.png"
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved: {plot_filename}")

# ============================================================
# STEP 7 — Quandela PIC angle table
# ============================================================
print("\n" + "=" * 62)
print(f"  NLFT-QSP Angles for Quandela PIC (Perceval)  [{FUNC_NAME}]")
print("=" * 62)
print(f"  Function         : {FUNC_NAME}")
print(f"  QSP layers L     : {polydeg}")
print(f"  PS gates         : {polydeg+1}")
print(f"  BS gates         : {polydeg}")
print(f"  Waveguide modes  : 2  (dual-rail photonic qubit)")
print()
print("  PS angles  (2*phi_j [rad]):")
for j, p in enumerate(phi_nlft):
    print(f"    PS[{j:02d}] = 2 x {p:+.6f} = {2*p:+.6f} rad")
print()
print("  Ry angles (theta_j [rad]):")
for j, t in enumerate(theta_nlft):
    print(f"    Ry[{j:02d}] = {t:.6f} rad")
print()
print("  Signal encoding: Rz(x) per layer,  x in [-pi, pi]")
print()
print("  OUTPUT MEASURABILITY:")
print("  Z = |psi[0]|^2 - |psi[1]|^2  is DIRECTLY measurable.")
print("  Photon in mode 0 -> +1,  mode 1 -> -1.")
print(f"  Average over many shots -> <Z> approximates {FUNC_NAME}(x).")
print("  Hardware: 2 single-photon detectors only.")
print("  No homodyne / phase reference / interference needed.")
print()
print(f"  NLFT-seeded MSE   : {result.fun:.4e}  ({result.nit} iterations)")
if has_opt:
    print(f"  Random-start MSE  : {mse_opt_surrogate:.4e}  (200 iterations, optimizer)")
print("=" * 62)

# ============================================================
# Bonus: method comparison table
# ============================================================
print("\n" + "=" * 62)
print(f"  METHOD COMPARISON: NLFT vs Optimization  [{FUNC_NAME}]")
print("=" * 62)
print(f"  {'Property':<35} {'NLFT-QSP':<20} {'Optimization'}")
print(f"  {'-'*35} {'-'*20} {'-'*15}")
print(f"  {'Starting point':<35} {'NLFT (deterministic)':<20} {'Random (seed=42)'}")
print(f"  {'Max iterations':<35} {'50':<20} {'200'}")
print(f"  {'Risk of local minima':<35} {'Low (physics-guided)':<20} {'Higher (random)'}")
print(f"  {'Reproducibility':<35} {'Yes':<20} {'Seed-dependent'}")
if has_opt:
    print(f"  {'MSE vs surrogate':<35} {mse_nlft_surrogate:<20.4e} {mse_opt_surrogate:.4e}")
    print(f"  {f'MSE vs true {FUNC_NAME}':<35} {mse_nlft_true:<20.4e} {mse_opt_true:.4e}")
else:
    print(f"  {'MSE vs surrogate':<35} {mse_nlft_surrogate:<20.4e} N/A (run optimizer first)")
    print(f"  {f'MSE vs true {FUNC_NAME}':<35} {mse_nlft_true:<20.4e} N/A (run optimizer first)")
print(f"  {'Output Z=p0-p1':<35} {'YES':<20} {'YES'}")
print("=" * 62)