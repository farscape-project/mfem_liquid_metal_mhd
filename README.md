# Liquid Metal MHD in MFEM

Repository for an inductionless liquid metal MHD solver in MFEM, detailed in [1] - deliverable for Fusion Computing Lab WP3.2.

The solver in `lmmhd_solver` contains `lmmhd_solver.cpp` which houses the `main()` function for the liquid metal magnetohydrodynamics (MHD) solver.  The solver currently solves the liquid metal MHD equations detailed in [1]. The solver sets up the following linear system for the coupled magnetohydrodynamics (MHD) problem:

      [  Mj    G^T    K^T       0   ] [ xj   ]   [ rj   ]
      [  G     0      0         0   ] [ xphi ]   [ rphi ]
      [ -K     0      Fk       B^T  ] [ xu   ] = [ ru   ]
      [  0     0      B         0   ] [ xp   ]   [ rp   ]

where the blocks are defined as:

- **Mj** : current-density bilinear form `(d, d')`
- **G**  : coupling between current and electric potential `-(div d, phi)`
- **K**  : coupling between current and velocity `(d, B × v')`
- **Fk** : velocity bilinear form `2/Tau * (v, v') + O(u*_n; v, v') + A_AL(v, v')`
- **B**  : coupling between velocity and pressure `-(div v, q)`

Work on the preconditioner [1] is required to improve solver efficiency and stability.  

The `lmmhd_solver` is run using the command, with options,

      ./lmmhd_solver -tf 0.5 -vs 5 -nx 8 -ny 10 -nz 10

where `ni` is number of elements in direction `i = x,y,z`, `tf` is final time, and `vs` is visualisation step.  Other options are `ci` and `li` for clustering intensity (not currently enabled) in direction `i = y,z` and domain size for `i = x,y,z`.


References:

[1] Li, Lingxiao, Mingjiu Ni, and Weiying Zheng. "A charge-conservative finite element method for inductionless MHD equations. Part II: a robust solver." SIAM Journal on Scientific Computing 41.4 (2019): B816-B842.
