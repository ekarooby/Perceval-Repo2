# ============================================================
# QSP STEP FUNCTION - QUANDELA QPU EXPERIMENTAL VERSION
# ============================================================
#
# GOAL:
#   Run the QSP STEP function circuit on the real Quandela
#   photonic chip via remote connection to Quandela Cloud.
#   This produces EXPERIMENTAL results from real hardware.
#
# DIFFERENCE FROM SLOS LOCAL CODE:
#   SLOS code   : pcvl.Processor("SLOS", circuit)
#                 runs locally on your computer, no internet,
#                 no token, no QPU credits consumed
#   This code   : pcvl.RemoteProcessor("qpu:belenos", token)
#                 submits job to real Quandela photonic chip,
#                 requires token + QPU credits,
#                 results are real experimental photon counts
#
# FILE NAMING CONVENTION:
#   Input angle files are loaded by ANGLE_L (set below).
#   ANGLE_L must match the L used when running nlft_qsp_step_anglefinder.py
#   Example: theta_step_nlft_L15.npy
#
#   All output files include FILE_TAG encoding function, L, N_shots, N_x:
#   Example (L=15, N=5000, x=25):
#     z_experimental_STEP_L15_N5000_x25.npy
#     qsp_experimental_STEP_L15_N5000_x25.png
#     job_ids_STEP_L15_N5000_x25.txt
#
#   SLOS comparison files are loaded by matching SLOS_TAG,
#   which must match the FILE_TAG used when running the SLOS code.
#   Example: z_slos_STEP_L15_N100000_x100.npy
#
# NAMING CONVENTIONS:
#   z_experimental       : Z = p0-p1 from real QPU hardware
#   f_perceval_analytic  : Perceval compute_unitary(), exact
#   z_slos               : local SLOS reference (if available)
#
# WHAT CHANGED FROM SLOS CODE:
#   1. Added token setup:      pcvl.RemoteConfig.set_token(...)
#   2. Replaced Processor:     pcvl.Processor("SLOS") ->
#                              pcvl.RemoteProcessor("qpu:belenos")
#   3. Added max_shots_per_call to Sampler (required for remote)
#   4. Changed job execution:  sample_count(N_SHOTS) ->
#                              sample_count.execute_async(N_SHOTS)
#   5. Added progress polling loop (async job, not instant)
#   6. Added job ID saving (so you can resume if disconnected)
#   7. Reduced x_values: 100 -> 25 (saves QPU credits)
#   8. All labels changed to "experimental" / "QPU"
#   9. Saved results use tagged filenames (no overwriting)
#   10. Added credit estimation before running
#
# HOW TO RUN:
#   1. Go to cloud.quandela.com and get your API token
#   2. Replace 'YOUR_API_TOKEN_HERE' with your actual token
#   3. Set ANGLE_L to match the angle file you want to load
#   4. Set SLOS_N and SLOS_X to match your SLOS run if comparing
#   5. Run: python qsp_step_experiment_qpu.py
#
# HOW TO RESUME A JOB IF DISCONNECTED:
#   If your connection drops mid-run, use the saved job IDs:
#   remote_proc = pcvl.RemoteProcessor("qpu:belenos")
#   job = remote_proc.resume_job("JOB_ID_FROM_job_ids_*.txt")
#   results = job.get_results()
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import perceval as pcvl
import perceval.components as comp
from perceval.algorithm import Sampler
import time
import os

# ============================================================
# Step 1: Token and QPU setup
# ============================================================

MY_TOKEN = "YOUR_API_TOKEN_HERE"   # <-- replace with your token
QPU_NAME = "qpu:belenos"           # <-- replace if using a different QPU

# ============================================================
# Step 2: Load QSP angles
#
# ANGLE_L : the L value of the angle file to load
#           must match the L used when running nlft_qsp_step_anglefinder.py
#           changing this is the ONLY thing needed to switch L
#
# theta_nlft : L+1 Ry rotation angles  (fixed, never change with x)
# phi_nlft   : L+1 Rz rotation angles  (fixed, never change with x)
# ============================================================

ANGLE_L = 15   # <-- change this to match the angle file you want to load

theta_opt = np.load(f"theta_step_nlft_L{ANGLE_L}.npy")
phi_opt   = np.load(f"phi_step_nlft_L{ANGLE_L}.npy")

L = len(theta_opt) - 1
print(f"Loaded QSP angles: L={L}")

# Sanity check: confirm loaded file L matches ANGLE_L
assert L == ANGLE_L, f"Mismatch: ANGLE_L={ANGLE_L} but loaded file has L={L}"

N_approx = 100

def step_surrogate(x):
    """Smooth arctan approximation of STEP function."""
    return (2.0 / np.pi) * np.arctan(N_approx * x)

def step_true(x):
    """Ideal STEP function: -1 for x<0, +1 for x>=0."""
    return np.where(x >= 0, 1.0, -1.0)

def build_qsp_pic(theta_arr, phi_arr, x_val, L):
    """
    Build the Perceval QSP circuit for a specific x value.
    Identical circuit structure to SLOS code -- no change.
    Structure per layer:
      Rz(phi_j) : PS(-phi_j/2) mode0 + PS(+phi_j/2) mode1  -- fixed
      Ry(theta_j): BS.Ry(theta_j)                           -- fixed
      Rz(x)     : PS(-x/2) mode0 + PS(+x/2) mode1          -- varies with x
    Perceval applies gates LEFT TO RIGHT so Rz is added before Ry.
    """
    circuit = pcvl.Circuit(2, name=f"QSP_STEP_L{L}")

    # Initial block A(theta_0, phi_0) = Ry(theta_0) * Rz(phi_0)
    circuit.add(0,      comp.PS(float(-phi_arr[0] / 2)))
    circuit.add(1,      comp.PS(float( phi_arr[0] / 2)))
    circuit.add((0, 1), comp.BS.Ry(theta=float(theta_arr[0])))

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
# Step 3: Experiment settings
#
# N_SHOTS  : shots per x value
# x_values : x points to sweep -- 25 is a good QPU balance
#            (100 for SLOS local, 25-30 for QPU to save credits)
#
# FILE_TAG : label embedded in ALL saved output filenames
#            encodes function name, L, N_SHOTS, N_X
#            ensures different runs never overwrite each other
#
# SLOS_TAG : label used to find the matching SLOS result files
#            for comparison in the 3-way plot.
#            Must match the FILE_TAG that was used when running
#            the SLOS simulation code.
#            Example: if SLOS was run with L=15, N=100000, x=100
#            then SLOS_TAG = "STEP_L15_N100000_x100"
#            Set to None to skip SLOS comparison.
# ============================================================

N_SHOTS  = 5000
x_values = np.linspace(-np.pi, np.pi, 25)
N_X      = len(x_values)

# Output file tag for this QPU run
FILE_TAG = f"STEP_L{L}_N{N_SHOTS}_x{N_X}"
print(f"\nFile tag for this run : {FILE_TAG}")

# SLOS tag to load for comparison -- set to None to skip
SLOS_TAG = f"STEP_L{L}_N100000_x100"   # <-- adjust N and x if your SLOS run used different values
                                        #     or set to None to skip SLOS comparison
print(f"SLOS tag for comparison: {SLOS_TAG}")

# ============================================================
# Step 4: Connect to Quandela Cloud QPU
# ============================================================

print("\n" + "=" * 60)
print("  Connecting to Quandela Cloud QPU")
print("=" * 60)

# Save token (only needs to be done once -- comment out after first run)
pcvl.RemoteConfig.set_token(MY_TOKEN)
pcvl.RemoteConfig().save()
print(f"  Token saved.")

# Connect to QPU
remote_proc_test = pcvl.RemoteProcessor(QPU_NAME)
specs = remote_proc_test.specs
print(f"  Connected to   : {QPU_NAME}")
print(f"  Max modes      : {specs['constraints']['max_mode_count']}")
print(f"  Max photons    : {specs['constraints']['max_photon_count']}")
print(f"  Our circuit    : 2 modes, 1 photon  -> OK")

# Estimate QPU credit cost before running
print(f"\n  Estimating QPU shots needed...")
circuit_test = build_qsp_pic(theta_opt, phi_opt, 0.0, L)
remote_proc_test.set_circuit(circuit_test)
remote_proc_test.with_input(pcvl.BasicState([1, 0]))
remote_proc_test.min_detected_photons_filter(1)
required_shots = remote_proc_test.estimate_required_shots(nsamples=N_SHOTS)
total_shots    = required_shots * N_X
print(f"  Shots per x point  : {required_shots:,}")
print(f"  x points           : {N_X}")
print(f"  Total shots needed : {total_shots:,}")
print(f"\n  >>> Check your QPU credit balance before proceeding <<<")
print("=" * 60)

# ============================================================
# Step 5: Run experimental sweep on QPU
#
# CHANGED FROM SLOS: the entire execution block is different.
#
# SLOS:   Sampler(local_proc)
#         results = sampler.sample_count(N_SHOTS)  [blocking, instant]
#
# QPU:    Sampler(remote_proc, max_shots_per_call=N_SHOTS)  [required!]
#         job = sampler.sample_count.execute_async(N_SHOTS) [non-blocking]
#         poll job.is_complete with time.sleep(5)           [wait loop]
#         results = job.get_results()
#
# WHY ASYNC:
#   QPU jobs are submitted to a queue. Your job may wait for
#   other users' jobs to finish first. The async pattern lets
#   your code wait politely without blocking your terminal.
#
# JOB IDs ARE SAVED:
#   If your connection drops, you can resume any job using
#   its ID. All job IDs are saved to job_ids_{FILE_TAG}.txt.
# ============================================================

z_experimental  = np.zeros(N_X)
p0_experimental = np.zeros(N_X)
p1_experimental = np.zeros(N_X)

# Job ID file tagged with FILE_TAG -- no overwriting between runs
job_id_file = f"job_ids_{FILE_TAG}.txt"
with open(job_id_file, "w") as f:
    f.write(f"QSP STEP QPU Experiment -- {QPU_NAME}\n")
    f.write(f"File tag : {FILE_TAG}\n")
    f.write(f"x points : {N_X}\n")
    f.write(f"N_SHOTS  : {N_SHOTS}\n\n")

print(f"\nStarting QPU experimental sweep:")
print(f"  {N_X} x values, {N_SHOTS} shots each")
print(f"  File tag   : {FILE_TAG}")
print(f"  Job IDs saved to: {job_id_file}")
print("=" * 60)

for i, x_val in enumerate(x_values):

    print(f"\n  [{i+1:2d}/{N_X}] x = {x_val:+.4f} rad")

    # Build circuit for this x value -- identical to SLOS
    circuit = build_qsp_pic(theta_opt, phi_opt, x_val, L)

    # CHANGED FROM SLOS: use RemoteProcessor instead of Processor("SLOS")
    remote_proc = pcvl.RemoteProcessor(QPU_NAME)
    remote_proc.set_circuit(circuit)
    remote_proc.with_input(pcvl.BasicState([1, 0]))
    remote_proc.min_detected_photons_filter(1)

    # CHANGED FROM SLOS: max_shots_per_call is REQUIRED for remote
    sampler = Sampler(remote_proc, max_shots_per_call=N_SHOTS)

    # CHANGED FROM SLOS: execute_async (non-blocking) instead of sample_count(N)
    job = sampler.sample_count.execute_async(N_SHOTS)

    # Save job ID immediately so we can resume if disconnected
    with open(job_id_file, "a") as f:
        f.write(f"x[{i:02d}] = {x_val:+.4f}  job_id = {job.id}\n")
    print(f"    Job submitted. ID: {job.id}")

    # CHANGED FROM SLOS: poll until job completes (QPU is async)
    print(f"    Waiting for QPU result", end="", flush=True)
    while not job.is_complete:
        time.sleep(5)
        print(".", end="", flush=True)
    print(f" done.")
    print(f"    Job status: {job.status()}")

    # Get results -- same structure as SLOS
    results = job.get_results()

    # Safety check: QPU sometimes returns None on real hardware
    if results is None or results.get('results') is None:
        print(f"    WARNING: no results returned for x={x_val:.4f} -- skipping")
        z_experimental[i]  = 0.0
        p0_experimental[i] = 0.0
        p1_experimental[i] = 0.0
        continue
    counts = dict(results['results'])

    count_mode0 = counts.get(pcvl.BasicState([1, 0]), 0)
    count_mode1 = counts.get(pcvl.BasicState([0, 1]), 0)
    total       = count_mode0 + count_mode1

    if total > 0:
        p0 = count_mode0 / total
        p1 = count_mode1 / total
        z  = p0 - p1
    else:
        p0, p1, z = 0.0, 0.0, 0.0
        print(f"    WARNING: no counts detected at x={x_val:.4f}")

    z_experimental[i]  = z
    p0_experimental[i] = p0
    p1_experimental[i] = p1

    print(f"    mode0={count_mode0:6d}  mode1={count_mode1:6d}  "
          f"p0={p0:.4f}  p1={p1:.4f}  Z={z:+.4f}")

print("\n" + "=" * 60)
print("Experimental sweep complete.")

# ============================================================
# Step 6: Save experimental results
#
# All filenames include FILE_TAG (function + L + N_shots + N_x)
# so each run saves to uniquely named files and never overwrites
# a previous run's results.
# ============================================================

np.save(f"x_values_{FILE_TAG}.npy",      x_values)
np.save(f"z_experimental_{FILE_TAG}.npy", z_experimental)
np.save(f"p0_experimental_{FILE_TAG}.npy", p0_experimental)
np.save(f"p1_experimental_{FILE_TAG}.npy", p1_experimental)
print(f"\nExperimental results saved with tag: {FILE_TAG}")

# ============================================================
# Step 7: Compute reference curves for comparison
#
# f_perceval_analytic : Perceval exact Z, no sampling, no QPU
#                       used to compare against experiment
# f_surrogate         : arctan target
# f_true              : ideal +-1 STEP
# z_slos              : SLOS local result (loaded by SLOS_TAG)
# ============================================================

x_fine      = np.linspace(-np.pi, np.pi, 300)
f_surrogate = np.array([step_surrogate(x) for x in x_fine])
f_true      = step_true(x_fine)

# Perceval analytic reference at same x points as experiment
print("\nComputing Perceval analytic reference...")
f_perceval_analytic = np.zeros(N_X)
for i, x_val in enumerate(x_values):
    circuit = build_qsp_pic(theta_opt, phi_opt, x_val, L)
    U   = np.array(circuit.compute_unitary())
    psi = U @ np.array([1.0, 0.0])
    f_perceval_analytic[i] = abs(psi[0])**2 - abs(psi[1])**2

# Load SLOS results for 3-way comparison (matched by SLOS_TAG)
slos_available = False
if SLOS_TAG is not None:
    slos_x_file = f"x_values_{SLOS_TAG}.npy"
    slos_z_file = f"z_slos_{SLOS_TAG}.npy"
    if os.path.exists(slos_x_file) and os.path.exists(slos_z_file):
        x_slos = np.load(slos_x_file)
        z_slos = np.load(slos_z_file)
        slos_available = True
        print(f"SLOS results loaded: {slos_z_file}")
    else:
        print(f"SLOS files not found ({slos_z_file}) -- skipping SLOS comparison.")
        print(f"  Run SLOS code first, or update SLOS_TAG to match your SLOS run.")

# ============================================================
# Step 8: MSE report
#
# mse_exp_vs_analytic : experimental vs Perceval analytic
#                       measures hardware noise + imperfections
#                       ideally small but will be nonzero (real chip)
#
# mse_exp_vs_surrogate: experimental vs arctan surrogate
#                       overall approximation quality
#
# mse_exp_vs_true     : experimental vs ideal STEP
#                       most physically meaningful number
# ============================================================

mse_exp_vs_analytic  = np.mean((z_experimental - f_perceval_analytic)**2)
mse_exp_vs_surrogate = np.mean((z_experimental - np.array([step_surrogate(x) for x in x_values]))**2)
mse_exp_vs_true      = np.mean((z_experimental - step_true(x_values))**2)

print(f"\n========== Experimental MSE Report  [{FILE_TAG}] ==========")
print(f"  MSE experimental vs Perceval analytic  : {mse_exp_vs_analytic:.4f}")
print(f"    (hardware noise + imperfections)")
print(f"  MSE experimental vs surrogate (arctan) : {mse_exp_vs_surrogate:.4f}")
print(f"  MSE experimental vs true STEP          : {mse_exp_vs_true:.4f}")
print(f"=====================================================")

# ============================================================
# Step 9: Plot experimental results
# ============================================================

ncols = 3 if slos_available else 2
fig, axes = plt.subplots(1, ncols, figsize=(6*ncols, 5))
fig.suptitle(
    f"QSP STEP Function -- Experimental Results  "
    f"QPU: {QPU_NAME}  L={L}  "
    f"x points={N_X}  N_shots={N_SHOTS}",
    fontsize=12, fontweight='bold'
)

xt = [-np.pi, 0, np.pi]
xl = [r"$-\pi$", r"$0$", r"$\pi$"]

# Left panel: experimental results
ax = axes[0]
ax.plot(x_fine,   f_true,              'k-',  lw=2.5,
        label="True STEP",                          zorder=3)
ax.plot(x_fine,   f_surrogate,         'g--', lw=1.5,
        label="arctan surrogate",                   zorder=2)
ax.plot(x_values, f_perceval_analytic, 'r-',  lw=1.5,
        label="Perceval analytic  Z=p0-p1",         zorder=4)
ax.plot(x_values, z_experimental,      'b.',  ms=10,
        label=f"Experimental  Z=p0-p1\n"
              f"  MSE vs surrogate={mse_exp_vs_surrogate:.4f}\n"
              f"  MSE vs true STEP={mse_exp_vs_true:.4f}",
        zorder=5)
ax.set_xlim([-np.pi, np.pi])
ax.set_ylim([-1.3, 1.3])
ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=11)
ax.set_xlabel(r"$x$", fontsize=12)
ax.set_ylabel("Z = p0 - p1", fontsize=12)
ax.set_title(f"Experimental QPU results  L={L}", fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Middle panel: residual experimental vs Perceval analytic
# Shows hardware noise (photon loss, imperfect BS, etc.)
ax2 = axes[1]
diff_exp = z_experimental - f_perceval_analytic
ax2.plot(x_values, diff_exp, color='darkred', lw=1.5, marker='.', ms=8,
         label=f"Experimental minus Perceval analytic\n"
               f"  MSE={mse_exp_vs_analytic:.4f}\n"
               f"  (hardware noise + imperfections)")
ax2.axhline(0, color='k', lw=0.8, linestyle='--')
ax2.fill_between(x_values, diff_exp, alpha=0.2, color='darkred')
ax2.set_xlim([-np.pi, np.pi])
ax2.set_xticks(xt); ax2.set_xticklabels(xl, fontsize=11)
ax2.set_xlabel(r"$x$", fontsize=12)
ax2.set_ylabel("Residual", fontsize=12)
ax2.set_title("Experimental vs Perceval analytic residual\n"
              "(shows real hardware noise)", fontsize=11)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Right panel: 3-way comparison (only if SLOS results loaded)
if slos_available:
    ax3 = axes[2]
    ax3.plot(x_fine,   f_true,              'k-',  lw=2.5,
             label="True STEP",                    zorder=3)
    ax3.plot(x_values, f_perceval_analytic, 'g--', lw=2,
             label="Perceval analytic",             zorder=4)
    ax3.plot(x_slos,   z_slos,              'b.',  ms=6,
             label=f"SLOS  ({SLOS_TAG})",           zorder=5)
    ax3.plot(x_values, z_experimental,      'r.',  ms=10,
             label=f"Experimental  ({FILE_TAG})",   zorder=6)
    ax3.set_xlim([-np.pi, np.pi])
    ax3.set_ylim([-1.3, 1.3])
    ax3.set_xticks(xt); ax3.set_xticklabels(xl, fontsize=11)
    ax3.set_xlabel(r"$x$", fontsize=12)
    ax3.set_title("Experimental vs SLOS vs Analytic\n"
                  "(3-way comparison)", fontsize=11)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

plt.tight_layout()

plot_filename = f"qsp_experimental_{FILE_TAG}.png"
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved: {plot_filename}")

# ============================================================
# SUMMARY OF CHANGES FROM SLOS CODE
# ============================================================
print("\n" + "=" * 60)
print("  SUMMARY OF CHANGES FROM SLOS LOCAL CODE")
print("=" * 60)
changes = [
    ("1",  "Token setup",
     "Added pcvl.RemoteConfig.set_token() + save()"),
    ("2",  "Processor",
     'pcvl.Processor("SLOS") -> pcvl.RemoteProcessor("qpu:belenos")'),
    ("3",  "Sampler",
     "Added max_shots_per_call=N_SHOTS (required for remote)"),
    ("4",  "Job execution",
     "sample_count(N) -> sample_count.execute_async(N)"),
    ("5",  "Wait loop",
     "Added polling loop while not job.is_complete"),
    ("6",  "Job ID saving",
     "Job IDs saved to job_ids_{FILE_TAG}.txt"),
    ("7",  "x_values",
     "100 points -> 25 points (saves QPU credits)"),
    ("8",  "Labels",
     "All labels now say 'experimental' / 'QPU'"),
    ("9",  "Saved files",
     "All .npy files include FILE_TAG in filename"),
    ("10", "Credit estimation",
     "Added estimate_required_shots() before running"),
    ("11", "3-way plot",
     "Right panel compares experimental vs SLOS vs analytic"),
    ("12", "SLOS loading",
     "SLOS files loaded by SLOS_TAG -- must match SLOS run tag"),
]
for num, name, desc in changes:
    print(f"  {num:>2}. {name:<25} {desc}")
print("=" * 60)