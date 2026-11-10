#pragma once
#include "mfem.hpp"
#include "parameters.hpp"

// Builds and owns the four FE spaces in the order the operator expects:
// 0 = j (RT), 1 = phi (L2), 2 = u (H1^dim), 3 = p (H1).
class Discretisation
{
public:
   Discretisation(mfem::ParMesh &pmesh, const lmmhd::DiscretisationParams &discPar)
      : currentD_fec(discPar.order_current_density, pmesh.Dimension()),
        electPot_fec(discPar.OrderPotential(), pmesh.Dimension()),
        velocity_fec(discPar.OrderVelocity(), pmesh.Dimension()),
        pressure_fec(discPar.order_pressure, pmesh.Dimension()),
        currentD_fespace(&pmesh, &currentD_fec),
        electPot_fespace(&pmesh, &electPot_fec),
        velocity_fespace(&pmesh, &velocity_fec, pmesh.Dimension()),
        pressure_fespace(&pmesh, &pressure_fec),
        spaces(4)
   {
      spaces[0] = &currentD_fespace;
      spaces[1] = &electPot_fespace;
      spaces[2] = &velocity_fespace;
      spaces[3] = &pressure_fespace;
   }

   mfem::Array<mfem::ParFiniteElementSpace *> &Spaces() { return spaces; }

private:
   // Collections must be declared before the spaces that use them.
   mfem::RT_FECollection currentD_fec;
   mfem::L2_FECollection electPot_fec;
   mfem::H1_FECollection velocity_fec;
   mfem::H1_FECollection pressure_fec;
   mfem::ParFiniteElementSpace currentD_fespace, electPot_fespace, velocity_fespace, pressure_fespace;
   mfem::Array<mfem::ParFiniteElementSpace *> spaces;
};