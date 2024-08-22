# mfem_liquid_metal_mhd

Repository for a liquid metal MHD code in MFEM - one of the FARSCAPE4 WP4.2 deliverables.

## Electrostatics

The script `01_electrostatic2D.cpp` solves the following equations
```math
\displaylines{k \vec{J} + \nabla \phi = \vec{f}, \\
- \nabla \cdot \vec{J} = g,}
```
for variables current density $J$ and electric scalar potential $\phi$ and coefficient $k = 1$ with functions $\vec{f} = (0,-1)$ and $g = 0$ with insulating boundary conditions on left and right walls and conducting boundary conditions on top and bottom.


## Electrostatics with $\vec{u} \times \vec{B}$

The script `02_electrostatic2DwithUcrossB.cpp` solves the following equations
```math
\displaylines{k \vec{J} + \nabla \phi + \vec{u} \times \vec{B} = \vec{f}, \\
- \nabla \cdot \vec{J} = g,}
```
in 3D for variables current density $J$ and electric scalar potential $\phi$ and coefficient $k = 1$ with functions $\vec{f} = (0,-1)$ and $g = 0$ with $u$ and $B$ constant. Boundary conditions are insulating on left and right walls and conducting on top and bottom.
