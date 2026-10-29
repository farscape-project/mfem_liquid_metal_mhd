# Plots the logged values (FGMRES residuals and relative L2 norms) from the
# MFEM liquid-metal MHD simulation. Values are saved by MFEM in simulation.log.

import re
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import sys

# Take the log file from the command line, or prompt for it if not given
if len(sys.argv) > 1:
    logfile = sys.argv[1]
else:
    logfile = input("Log file to plot (e.g. stdout.472084): ").strip()

if not os.path.isfile(logfile):
    sys.exit(f"Error: '{logfile}' not found")

# ------------------------------------------------------------
# Storage
# ------------------------------------------------------------

residuals = []  # (timestep, pass, local_iter, residual)
l2_timesteps = defaultdict(list)

global_iter = 0
timestep = -1
timestep_boundaries = []  # (global_iter at start of step, timestep)

dirname = "plots/residuals/"
os.makedirs(dirname, exist_ok=True)

# ------------------------------------------------------------
# Regex patterns
# ------------------------------------------------------------

step_re = re.compile(r"Time step (\d+), time = ([0-9.eE+-]+)")
# Matches e.g. "   Pass :  2   Iteration :  51  || r || = 5.72468e-05"
iter_re = re.compile(
    r"Pass\s*:\s*(\d+)\s+Iteration\s*:\s*(\d+)\s+\|\|\s*r\s*\|\|\s*=\s*([0-9.eE+-]+)"
)
l2_re = re.compile(r"Relative L2 Norm for (.+?):\s*([0-9.eE+-]+)")

# ------------------------------------------------------------
# Parse file
# ------------------------------------------------------------

with open(logfile, "r") as f:
    for line in f:

        m = step_re.search(line)
        if m:
            timestep = int(m.group(1))
            timestep_boundaries.append((global_iter, timestep))
            continue

        m = iter_re.search(line)
        if m:
            p = int(m.group(1))
            it = int(m.group(2))
            r = float(m.group(3))
            residuals.append((timestep, p, it, r))
            global_iter += 1
            continue

        m = l2_re.search(line)
        if m:
            field = m.group(1).strip()
            val = float(m.group(2))
            l2_timesteps[field].append((timestep, val))
            continue

if not residuals:
    sys.exit("No FGMRES residuals found in the log - check the iteration line format.")

# ------------------------------------------------------------
# 0. Helpers to make plots look nice
# ------------------------------------------------------------

def add_verticals_at_time_steps(timestep_boundaries):
    for xpos, ts in timestep_boundaries:
        plt.axvline(x=xpos, color='gray', linestyle='--', linewidth=0.75, alpha=0.5)


def add_timestep_boundaries_as_2nd_axis(timestep_boundaries):
    ax = plt.gca()
    ax_top = ax.secondary_xaxis('top')
    ax_top.set_xticks([p for p, ts in timestep_boundaries])
    ax_top.set_xticklabels([ts for p, ts in timestep_boundaries])
    ax_top.set_xlabel("Time step")

# ------------------------------------------------------------
# 1. FGMRES residual plot
# ------------------------------------------------------------

plt.figure()
add_verticals_at_time_steps(timestep_boundaries)
plt.semilogy(range(len(residuals)), [r[3] for r in residuals])
plt.xlabel("Cumulative iteration")
add_timestep_boundaries_as_2nd_axis(timestep_boundaries)
plt.ylabel("||r||")
plt.title("FGMRES Residual History")
plt.grid(True, axis='y', linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(dirname, "gmres_residual.png"), dpi=300)
plt.close()

# ------------------------------------------------------------
# 2. L2 norms per timestep
# ------------------------------------------------------------

if l2_timesteps:
    plt.figure()
    for field, data in l2_timesteps.items():
        x = [d[0] for d in data]
        y = [d[1] for d in data]
        plt.semilogy(x, y, marker='o', label=field)

    plt.xlabel("Time step")
    plt.ylabel("Relative L2 Norm")
    plt.title("Field Convergence per Time Step")
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(dirname, "l2_norms.png"), dpi=300)
    plt.close()

print(f"Parsed {len(residuals)} FGMRES iterations over "
      f"{len(timestep_boundaries)} time steps; plots saved to {dirname}")