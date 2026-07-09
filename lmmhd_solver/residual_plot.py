# Plots the logged values (residuals and preconditioner norms) from the 
# MFEM liquid-metal MHD simulation. Values are saved by MFEM in simulation.log.

import re
import matplotlib.pyplot as plt
from collections import defaultdict
import os

logfile = "simulation.log"

# ------------------------------------------------------------
# Storage
# ------------------------------------------------------------

residuals = []  # (timestep, local_iter, residual)
precond_norms = defaultdict(list)
l2_timesteps = defaultdict(list)
field_norms = defaultdict(list)

global_iter = 0
timestep = -1

dirname = "plots/residuals/"
os.makedirs(dirname, exist_ok=True)

timestep_boundaries = []

# ------------------------------------------------------------
# Regex patterns
# ------------------------------------------------------------

step_re = re.compile(r"Time step (\d+), time = ([0-9.eE+-]+)")
iter_re = re.compile(r"FGMRES iteration\s*:\s*(\d+).*?\|\|\s*r\s*\|\|\s*=\s*([0-9.eE+-]+)")
norm_re = re.compile(r"\|\|(\w+)\|\|\s*=\s*([0-9.eE+-]+)")
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
            it = int(m.group(1))
            r = float(m.group(2))
            residuals.append((timestep, it, r))
            global_iter += 1
            continue

        for name, val in norm_re.findall(line):
            precond_norms[name].append((global_iter, float(val)))

        m = l2_re.search(line)
        if m:
            field = m.group(1).strip()
            val = float(m.group(2))
            l2_timesteps[field].append((timestep, val))
            continue

        # = field_norm_re.search(line)
        #if m:
        #    field = m.group(1)
        #    val = float(m.group(2))
        #    field_norms[field].append((timestep, val))



# ------------------------------------------------------------
# 0. Functions to make plots look nice.
# ------------------------------------------------------------

def add_verticals_at_time_steps(timestep_boundaries):

    # Vertical lines marking timestep boundaries
    for xpos, ts in timestep_boundaries[:-1]:
        plt.axvline(
            x=xpos,
            color='gray',
            linestyle='--',
            linewidth=0.75,
            alpha=0.5
        )
    return

def add_timestep_boundaries_as_2nd_axis(timestep_boundaries):

    ax = plt.gca()

    # Add label on top axis
    ax_top = ax.secondary_xaxis('top')
    # Put ticks where timesteps begin
    tick_positions = [p for p, ts in timestep_boundaries[:-1]]
    tick_labels = [ts for p, ts in timestep_boundaries[:-1]]
    ax_top.set_xticks(tick_positions)
    ax_top.set_xticklabels(tick_labels)

    ax_top.set_xlabel("Time step")

    return


# ------------------------------------------------------------
# 1. FGMRES residual plot
# ------------------------------------------------------------

plt.figure()
add_verticals_at_time_steps(timestep_boundaries)
plt.semilogy([r[2] for r in residuals])
plt.xlabel("Iteration")
add_timestep_boundaries_as_2nd_axis(timestep_boundaries)
plt.ylabel("||r||")
plt.title("FGMRES Residual History")
plt.grid(True, axis='y', linestyle='--')
plt.tight_layout()
plt.savefig(''.join([dirname,"gmres_residual.png"]), dpi=300)
plt.close()

# ------------------------------------------------------------
# 2a. Residual-type norms (r family)
# ------------------------------------------------------------

plt.figure()
add_verticals_at_time_steps(timestep_boundaries)

for name, data in precond_norms.items():
    if not name.startswith("r"):
        continue

    x = [d[0] for d in data]
    y = [d[1] for d in data]
    plt.plot(x, y, label=name)

plt.yscale("log")
plt.xlabel("Iteration")
add_timestep_boundaries_as_2nd_axis(timestep_boundaries)
plt.ylabel("Norm value")
plt.title("Residual-type norms (r family)")
plt.legend()
plt.grid(True, axis='y', linestyle='--')
plt.tight_layout()
plt.savefig(''.join([dirname,"preconditioner_norms_r.png"]), dpi=300)
plt.close()

# ------------------------------------------------------------
# 2b. Solution-update norms (y family)
# ------------------------------------------------------------

plt.figure()
add_verticals_at_time_steps(timestep_boundaries)

for name, data in precond_norms.items():
    if not name.startswith("y"):
        continue

    x = [d[0] for d in data]
    y = [d[1] for d in data]
    plt.plot(x, y, label=name)

plt.yscale("log")
plt.xlabel("Iteration")
add_timestep_boundaries_as_2nd_axis(timestep_boundaries)
plt.ylabel("Norm value")
plt.title("Solution-update norms (y family)")
plt.legend()
plt.grid(True, axis='y', linestyle='--')
plt.tight_layout()
plt.savefig(''.join([dirname,"preconditioner_norms_y.png"]), dpi=300)
plt.close()

# ------------------------------------------------------------
# 2c. Other internal norms
# ------------------------------------------------------------

plt.figure()
add_verticals_at_time_steps(timestep_boundaries)

for name, data in precond_norms.items():
    if name.startswith("r") or name.startswith("y"):
        continue

    x = [d[0] for d in data]
    y = [d[1] for d in data]
    plt.plot(x, y, label=name)

plt.yscale("log")
plt.xlabel("Iteration")
add_timestep_boundaries_as_2nd_axis(timestep_boundaries)
plt.ylabel("Norm value")
plt.title("Other preconditioner norms")
plt.legend()
plt.grid(True, axis='y', linestyle='--')
plt.tight_layout()
plt.savefig(''.join([dirname,"preconditioner_norms_other.png"]), dpi=300)
plt.close()

# ------------------------------------------------------------
# 3. L2 norms per timestep
# ------------------------------------------------------------

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
plt.savefig(''.join([dirname,"l2_timesteps.png"]), dpi=300)
plt.close()