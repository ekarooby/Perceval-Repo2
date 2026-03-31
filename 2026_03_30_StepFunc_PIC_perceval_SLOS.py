# ============================================================
# QSP LOCAL SLOS SIMULATION -- STEP / ReLU / SELU
# ============================================================
#
# GOAL:
#   Simulate the QSP circuit locally using Perceval's SLOS
#   backend, which fires photons one by one and counts
#   detections -- mimicking a real experiment.
#
# DIFFERENCE FROM CIRCUIT CODE:
#   Circuit code: uses compute_unitary() -- exact matrix math
#   This code:    uses SLOS sampler -- statistical photon counting
#
# HOW TO SWITCH BETWEEN STEP / ReLU / SELU:
#   Change FUNC_NAME and ANGLE_L at the top of the settings
#   block below. Everything else -- angle file loaded, surrogate
#   function, file names, plot titles -- updates automatically.
#   FUNC_NAME = "STEP"   --> loads theta_step_nlft_L{ANGLE_L}.npy
#   FUNC_NAME = "ReLU"   --> loads theta_relu_nlft_L{ANGLE_L}.npy
#   FUNC_NAME = "SELU"   --> loads theta_selu_nlft_L{ANGLE_L}.npy
#
# NOTE ON N_SHOTS:
#   N_SHOTS = 100000 is sufficient. Do not increase it.
#   MSE SLOS vs Perceval analytic = ~0 --> sampling noise negligible
#   Increasing N_SHOTS will NOT improve approximation quality.
#   To improve approximation quality:
#     --> increase L (more QSP layers), NOT N_SHOTS
#
# NOTE ON X POINTS:
#   x_values = 100 points used for local SLOS simulation
#   --> gives smooth curve, fast locally, free to run
#   For real QPU run, reduce to 30 points to save QPU credits
#   MSE vs true function depends on number of x points:
#     30 points  --> MSE ~ 0.0775
#     100 points --> MSE ~ 0.0476
#   This is NOT because the circuit improved -- it is because
#   more x points gives a more representative average over [-pi, pi]
#
# FILE NAMING CONVENTION:
#   All output files are labeled with FUNC_NAME, L, N_shots, N_x
#   so that runs with different parameters never overwrite each other,
#   and STEP / ReLU / SELU results stay clearly separated.
#   Example (FUNC_NAME="STEP", L=15, N=100000, x=100):
#     z_slos_STEP_L15_N100000_x100.npy
#     qsp_slos_STEP_L15_N100000_x100.png
#
# NAMING CONVENTIONS:
#   f_classical          : pure numpy matrix math, no Perceval (circuit code)
#   f_perceval_analytic  : Perceval compute_unitary(), exact, no sampling
#   z_slos               : Perceval SLOS sampler, statistical photon counts
#   z_experimental       : reserved for future real QPU hardware results
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import perceval as pcvl
import perceval.components as comp
from perceval.algorithm import Sampler

# ============================================================
# ---- SETTINGS: change these two lines to switch function ----
#
# FUNC_NAME : "STEP", "ReLU", or "SELU"
#             controls which angle file is loaded, which surrogate
#             is used, and all output file names and plot titles
# ANGLE_L   : must match the L used when running nlft_qsp_anglefinder.py
#             changing this is the only thing needed to switch L
# ============================================================

FUNC_NAME = "STEP"   # <-- change to "ReLU" or "SELU" as needed
ANGLE_L   = 100 # <-- change to match the angle file you want to load

# ============================================================
# Derived settings -- do not change these manually
# ============================================================

FUNC_LOWER = FUNC_NAME.lower()   # "step", "relu", "selu" -- used in filenames
N_approx   = 100                 # sharpness of arctan surrogate (used for STEP)

# ============================================================
# Step 1: Load QSP angles
#
# Angle files are named by FUNC_LOWER and ANGLE_L, matching
# exactly what nlft_qsp_anglefinder.py saved.
# theta : L+1 Ry rotation angles  (fixed, never change with x)
# phi   : L+1 Rz rotation angles  (fixed, never change with x)
# ============================================================

theta_nlft = np.load(f"theta_{FUNC_LOWER}_nlft_L{ANGLE_L}.npy")
phi_nlft   = np.load(f"phi_{FUNC_LOWER}_nlft_L{ANGLE_L}.npy")

L = len(theta_nlft) - 1
print(f"Loaded QSP angles: FUNC={FUNC_NAME}  L={L}")

# Sanity check: confirm loaded file L matches ANGLE_L
assert L == ANGLE_L, f"Mismatch: ANGLE_L={ANGLE_L} but loaded file has L={L}"

# ============================================================
# Target functions
#
# Each FUNC_NAME has its own surrogate (smooth) and true function.
# surrogate_func : used as optimization target and for MSE reporting
# true_func      : ideal target, used for MSE reporting only
# ============================================================

def get_surrogate(func_name):
    """Return the smooth surrogate function for the given FUNC_NAME."""
    if func_name == "STEP":
        return lambda x: (2.0 / np.pi) * np.arctan(N_approx * x)
    elif func_name == "ReLU":
        return lambda x: np.log(1 + np.exp(N_approx * x)) / N_approx
    elif func_name == "SELU":
        alpha = 1.6733
        scale = 1.0507
        return lambda x: scale * np.where(x >= 0, x, alpha * (np.exp(x) - 1))
    else:
        raise ValueError(f"Unknown FUNC_NAME: {func_name}. Use 'STEP', 'ReLU', or 'SELU'.")

def get_true_func(func_name):
    """Return the ideal target function for the given FUNC_NAME."""
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

def build_qsp_pic(theta_arr, phi_arr, x_val, L):
    """
    Build the Perceval QSP circuit for a specific x value.
    Structure per layer:
      Rz(phi_j) : PS(-phi_j/2) mode0 + PS(+phi_j/2) mode1  -- fixed
      Ry(theta_j): BS.Ry(theta_j)                           -- fixed
      Rz(x)     : PS(-x/2) mode0 + PS(+x/2) mode1          -- changes with x
    Perceval applies gates LEFT TO RIGHT so Rz is added before Ry.
    """
    circuit = pcvl.Circuit(2, name=f"QSP_{FUNC_NAME}_L{L}")

    # Initial block A(theta_0, phi_0) = Ry(theta_0) * Rz(phi_0)
    circuit.add(0,      comp.PS(float(-phi_arr[0] / 2)))   # Rz first
    circuit.add(1,      comp.PS(float( phi_arr[0] / 2)))
    circuit.add((0, 1), comp.BS.Ry(theta=float(theta_arr[0])))  # then Ry

    for j in range(1, L + 1):
        # Signal unitary Rz(x) -- only part that changes with x
        circuit.add(0, comp.PS(float(-x_val / 2)))
        circuit.add(1, comp.PS(float( x_val / 2)))
        # Fixed block A(theta_j, phi_j) = Ry(theta_j) * Rz(phi_j)
        circuit.add(0,      comp.PS(float(-phi_arr[j] / 2)))
        circuit.add(1,      comp.PS(float( phi_arr[j] / 2)))
        circuit.add((0, 1), comp.BS.Ry(theta=float(theta_arr[j])))

    return circuit

# ============================================================
# Step 2: Experiment settings
#
# N_SHOTS  : number of photons fired per x value
#            100000 is sufficient -- see note at top of file
# x_values : x points to sweep over [-pi, pi]
#            100 points locally, reduce to 30 for QPU runs
#
# FILE_TAG : label embedded in all saved filenames
#            encodes FUNC_NAME, L, N_SHOTS, N_X
#            ensures different runs never overwrite each other
# ============================================================

N_SHOTS  = 5000
x_values = np.linspace(-np.pi, np.pi, 100)
N_X      = len(x_values)

FILE_TAG = f"{FUNC_NAME}_L{L}_N{N_SHOTS}_x{N_X}"
print(f"\nFile tag for this run: {FILE_TAG}")

# ============================================================
# Step 3: Run SLOS local simulation
#
# For each x value:
#   1. Build circuit with that x baked into signal PS phases
#   2. Run SLOS sampler -- fires N_SHOTS photons through circuit
#   3. Count detections in mode 0 and mode 1
#   4. Compute p0 = count_mode0 / total
#              p1 = count_mode1 / total
#              Z  = p0 - p1
#
# SLOS = Scalable Linear Optical Simulator
# Local backend -- no token, no internet, runs instantly
# Models probabilistic photon detection like real hardware
# ============================================================

z_slos  = np.zeros(N_X)
p0_slos = np.zeros(N_X)
p1_slos = np.zeros(N_X)

print(f"\nStarting SLOS sweep:")
print(f"  Function : {FUNC_NAME}")
print(f"  {N_X} x values, {N_SHOTS} shots each")
print(f"  Total photons fired: {N_X * N_SHOTS:,}")
print("=" * 55)

for i, x_val in enumerate(x_values):

    circuit    = build_qsp_pic(theta_nlft, phi_nlft, x_val, L)
    local_proc = pcvl.Processor("SLOS", circuit)
    local_proc.with_input(pcvl.BasicState([1, 0]))
    local_proc.min_detected_photons_filter(1)

    sampler = Sampler(local_proc)
    results = sampler.sample_count(N_SHOTS)
    counts  = dict(results['results'])

    count_mode0 = counts.get(pcvl.BasicState([1, 0]), 0)
    count_mode1 = counts.get(pcvl.BasicState([0, 1]), 0)
    total       = count_mode0 + count_mode1

    if total > 0:
        p0 = count_mode0 / total
        p1 = count_mode1 / total
        z  = p0 - p1
    else:
        p0, p1, z = 0.0, 0.0, 0.0
        print(f"  WARNING: no counts detected at x={x_val:.3f}")

    z_slos[i]  = z
    p0_slos[i] = p0
    p1_slos[i] = p1

    print(f"  [{i+1:3d}/{N_X}] x={x_val:+.3f}  "
          f"mode0={count_mode0}  mode1={count_mode1}  "
          f"p0={p0:.3f}  p1={p1:.3f}  Z={z:+.3f}")

print("=" * 55)
print("Sweep complete.")

# ============================================================
# Step 4: Save SLOS results to disk
#
# All filenames include FILE_TAG (FUNC_NAME + L + N + x)
# so results from different functions, L, N_shots, or x-count
# runs are kept separate and never overwrite each other.
# ============================================================

np.save(f"x_values_{FILE_TAG}.npy", x_values)
np.save(f"z_slos_{FILE_TAG}.npy",   z_slos)
np.save(f"p0_slos_{FILE_TAG}.npy",  p0_slos)
np.save(f"p1_slos_{FILE_TAG}.npy",  p1_slos)
print(f"\nSLOS results saved with tag: {FILE_TAG}")

# ============================================================
# Step 5: Compute reference curves for comparison
#
# f_perceval_analytic : Perceval circuit + compute_unitary()
#                       exact Z via linear algebra, no sampling
#                       used to verify SLOS sampling is accurate
# f_surrogate         : smooth surrogate we optimized against
# f_true              : ideal target function
# ============================================================

x_fine      = np.linspace(-np.pi, np.pi, 300)
f_surrogate = surrogate_func(x_fine)
f_true      = true_func(x_fine)

# Perceval analytic Z at same x points as SLOS sweep
# Uses compute_unitary() -- exact linear algebra, no randomness
f_perceval_analytic = np.zeros(N_X)
for i, x_val in enumerate(x_values):
    circuit = build_qsp_pic(theta_nlft, phi_nlft, x_val, L)
    U       = np.array(circuit.compute_unitary())
    psi     = U @ np.array([1.0, 0.0])
    f_perceval_analytic[i] = abs(psi[0])**2 - abs(psi[1])**2

# ============================================================
# Step 6: MSE report
#
# mse_slos_vs_analytic  : SLOS vs Perceval analytic
#                         measures pure sampling noise
#                         should be ~0 with N_SHOTS=100000
#
# mse_slos_vs_surrogate : SLOS vs smooth surrogate
#                         measures how well the circuit fits
#                         the function it was trained on
#
# mse_slos_vs_true      : SLOS vs ideal true function
#                         most physically meaningful number
#                         = QSP approximation error + sampling noise
# ============================================================

mse_slos_vs_analytic  = np.mean((z_slos - f_perceval_analytic)**2)
mse_slos_vs_surrogate = np.mean((z_slos - surrogate_func(x_values))**2)
mse_slos_vs_true      = np.mean((z_slos - true_func(x_values))**2)

print(f"\n========== MSE Report  [{FILE_TAG}] ==========")
print(f"  MSE SLOS vs Perceval analytic : {mse_slos_vs_analytic:.4f}  must be ~0")
print(f"  MSE SLOS vs surrogate         : {mse_slos_vs_surrogate:.4f}")
print(f"  MSE SLOS vs true {FUNC_NAME:<5}        : {mse_slos_vs_true:.4f}")
print(f"==============================================")

# ============================================================
# Step 7: Plot results
#
# Left panel:  SLOS sampled Z vs Perceval analytic vs true function
#              blue dots label shows BOTH MSE vs surrogate
#              AND MSE vs true function
#              blue dots should sit on red line
# Right panel: residual SLOS minus Perceval analytic
#              shows sampling noise -- should be close to zero
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    f"QSP {FUNC_NAME} - SLOS Local Simulation  "
    f"L={L}  N_shots={N_SHOTS}  N_x={N_X}",
    fontsize=13
)

xt = [-np.pi, 0, np.pi]
xl = [r"$-\pi$", r"$0$", r"$\pi$"]

# Left panel
ax = axes[0]
ax.plot(x_fine,   f_true,              'k-',  lw=2,
        label=f"True {FUNC_NAME}")
ax.plot(x_fine,   f_surrogate,         'g--', lw=1.5,
        label="surrogate")
ax.plot(x_values, f_perceval_analytic, 'r-',  lw=1.5,
        label="Perceval analytic Z=p0-p1")
ax.plot(x_values, z_slos,              'b.',  ms=8,
        label=f"SLOS sampled Z=p0-p1\n"
              f"  MSE vs surrogate={mse_slos_vs_surrogate:.4f}\n"
              f"  MSE vs true {FUNC_NAME}={mse_slos_vs_true:.4f}")
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-1.3, 1.3])
ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=12)
ax.set_xlabel(r"$x$", fontsize=12)
ax.set_ylabel("Z = p0 - p1", fontsize=12)
ax.set_title(f"{FUNC_NAME}  L={L}  blue dots must sit on red line",
             fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right panel: residual between SLOS and Perceval analytic
# Shows pure sampling noise -- should be close to zero
ax2 = axes[1]
diff = z_slos - f_perceval_analytic
ax2.plot(x_values, diff, 'purple', lw=1.5, marker='.', ms=6,
         label=f"SLOS minus Perceval analytic  "
               f"MSE={mse_slos_vs_analytic:.4f}")
ax2.axhline(0, color='k', lw=0.8, linestyle='--')
ax2.fill_between(x_values, diff, alpha=0.2, color='purple')
ax2.set_xlim([-np.pi, np.pi])
ax2.set_xticks(xt); ax2.set_xticklabels(xl, fontsize=12)
ax2.set_xlabel(r"$x$", fontsize=12)
ax2.set_ylabel("Residual", fontsize=12)
ax2.set_title(f"SLOS vs Perceval analytic residual  ({FUNC_NAME})\n"
              "(shows sampling noise -- should be ~0)", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()

plot_filename = f"qsp_slos_{FILE_TAG}.png"
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved: {plot_filename}")