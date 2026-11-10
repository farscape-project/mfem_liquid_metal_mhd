#pragma once
#include "mfem.hpp"
#include <array>
#include <cmath>

namespace lmmhd
{

struct PhysicsParams
{
   mfem::real_t Re = 1.0;                       // Reynolds number
   mfem::real_t kappa = 1.0;                    // Coupling number N = Ha^2 / Re
   mfem::real_t alpha = 1.0;                    // Grad-div (augmented Lagrangian) parameter
   std::array<mfem::real_t, 3> B = {0.0, 0.0, 1.0};  // Applied magnetic field

   mfem::real_t reciprocalRe() const { return 1.0 / Re; }
   mfem::real_t alpha1() const       { return alpha + reciprocalRe(); }
   mfem::real_t Ha() const           { return std::sqrt(kappa * Re); }
};

struct TimeParams
{
   mfem::real_t dt = 0.01;     // Fixed: not currently adaptable.
   mfem::real_t theta = 0.6;   // Theta-scheme parameter (0.5 to match Crank-Nicolson scheme in Li et al. (2019)).
};

struct LinearSolverParams
{
   mfem::real_t rel_tol = 1e-8;
   mfem::real_t abs_tol = 1e-12;
   int max_iter = 300;
   int kdim = 50;
   bool use_ads_for_current_density = true;  // false -> MUMPS for the j block
   int debug = 0;                            // > 0 enables preconditioner norm logging
};

struct DiscretisationParams
{
   // RT_k / L2_k for (j, phi) and Taylor-Hood H1_{p+1} / H1_p for (u, p).
   int order_current_density = 0;
   int order_pressure = 1;

   int OrderPotential() const { return order_current_density; }
   int OrderVelocity() const  { return order_pressure + 1; }
};

struct Parameters
{
   PhysicsParams physics;
   TimeParams time;
   LinearSolverParams solver;
   DiscretisationParams discretisation;
};

} // namespace lmmhd