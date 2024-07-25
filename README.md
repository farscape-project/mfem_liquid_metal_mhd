# mfem_liquid_metal_mhd

Repository for a liquid metal MHD code in MFEM - one of the FARSCAPE4 WP4.2 deliverables.

## Current Solve

The script `01_current_solve.cpp` solves the following equations
```math
\displaylines{k \vec{J} + \nabla \phi = \vec{f}, \\
- \nabla \cdot \vec{J} = g,}
```
for variables current density $J$ and electric scalar potential $\phi$ and coefficient $k = 1$ with functions $\vec{f} = \vec{0}$ and $g = 0$.
