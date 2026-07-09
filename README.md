# Liquid Metal MHD in MFEM

Repository for an inductionless liquid metal MHD solver in MFEM, detailed in [1].

The solver is built out of `lmmhd_solver.cpp` which houses the `main()` function for the liquid metal magnetohydrodynamics (MHD) solver.  The solver currently solves the liquid metal MHD equations detailed in [1]. The solver sets up the following linear system for the coupled magnetohydrodynamics (MHD) problem:

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

The `lmmhd_solver` is run using the command, with options,

      ./lmmhd_solver

where solver options and parameters are specified in the `params.in` file.


References:

[1] Li, Lingxiao, Mingjiu Ni, and Weiying Zheng. "A charge-conservative finite element method for inductionless MHD equations. Part II: a robust solver." SIAM Journal on Scientific Computing 41.4 (2019): B816-B842.
