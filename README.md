# Liquid Metal MHD in MFEM

Repository for an inductionless liquid metal MHD solver in MFEM, detailed in [1].

The solver is built out of `lmmhd_solver.cpp` which houses the `main()` function for the liquid metal magnetohydrodynamics (MHD) solver.  The solver currently solves the liquid metal MHD equations detailed in [1]. The solver sets up the following linear system for the coupled magnetohydrodynamics (MHD) problem:

      [  Mj    G^T    K^T       0   ] [ xj   ]   [ rj   ]
      [  G     0      0         0   ] [ xphi ]   [ rphi ]
      [ -κK    0      Fu       B^T  ] [ xū   ] = [ ru   ]
      [  0     0      B         0   ] [ xp   ]   [ rp   ]

where κ = Ha²/Re, ū = (u^{n+1} + u^n)/2 is the midpoint velocity (so J, φ and p are at t^{n+1/2}), u* = (3u^n − u^{n−1})/2, and the blocks are defined as:

- **Mj** : current-density bilinear form `(d, d')`
- **G**  : coupling between current and electric potential `-(div d, phi)`
- **K**  : coupling between current and velocity `(d, B × v')`
- **Fu** : velocity bilinear form `2/τ (v, v') + O(u*; v, v') + A_AL(v, v')`, with the skew-symmetric convection form `O(w; v, v') = ½(w·∇v, v') − ½(w·∇v', v)` and `A_AL(v, v') = 1/Re (∇v, ∇v') + α (div v, div v')`
- **B**  : coupling between velocity and pressure `-(div v, q)`

The current density and potential use matching RT_k × L2_k spaces, so the discrete current is exactly divergence-free (charge-conservative); ‖div J_h‖ is reported every step. The velocity/pressure pair is Taylor–Hood (Q2–Q1 on hexahedra).

## Building

Requires a parallel MFEM build (MPI + hypre; MUMPS for the default current-density solver).

      make MFEM_DIR=/path/to/mfem MFEM_LIB_SUFFIX=        # standard MFEM build
      make                                                # MOOSE-bundled MFEM (libmfem-opt)
      make BUILD=debug ...                                # -g -O0 with AddressSanitizer
      make test ...                                       # unit test: convection operator is skew-symmetric

Use release builds for any timing or iteration-count study.

## Running

      mpirun -np 4 ./lmmhd_solver [options]

Parameters are read from `params.in`; command-line options (`./lmmhd_solver -h`) override them. Main parameters:

| parameter | meaning |
|---|---|
| `nx ny nz`, `Lx Ly Lz` | number of elements and domain size |
| `clusterX/Y/Z` | tanh clustering of vertices towards the walls |
| `t_final`, `dt`, `vis_steps` | time stepping and ParaView output frequency |
| `Re`, `Ha`, `Bx By Bz` | Reynolds and Hartmann numbers, imposed field direction |
| `alpha` | grad-div (augmented Lagrangian) parameter; `0` disables it |
| `dj_solver` | current-density block of the preconditioner: `mumps` (direct) or `ads` (5 CG iterations with Hiptmair–Xu, as in Algorithm 4.1 of [1]) |

Output is written to `data/lmmhd` (ParaView) and `simulation.log`. Cycle n holds u at t = n·dt and J, φ, p at t = (n − ½)·dt.

References:

[1] Li, Lingxiao, Mingjiu Ni, and Weiying Zheng. "A charge-conservative finite element method for inductionless MHD equations. Part II: a robust solver." SIAM Journal on Scientific Computing 41.4 (2019): B816-B842.
