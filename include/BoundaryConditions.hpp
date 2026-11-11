#pragma once

#include "mfem.hpp"
#include <memory>
#include <string>
#include <initializer_list>

// Boundary attributes produced by Mesh::MakeCartesian3D.
namespace BoxFace
{
   enum : int { ZMin = 1, YMin = 2, XMax = 3, YMax = 4, XMin = 5, ZMax = 6 };
}

struct BoundaryConditions
{
   std::string description;

   // Essential-boundary markers: one entry per boundary attribute, 1 = essential.
   mfem::Array<int> currentD_ess, electPot_ess, velocity_ess, pressure_ess;

   // Dirichlet data for the velocity on velocity_ess.
   std::unique_ptr<mfem::VectorCoefficient> velocity;

   explicit BoundaryConditions(const mfem::ParMesh &pmesh)
   {
      const int n = pmesh.bdr_attributes.Size() ? pmesh.bdr_attributes.Max() : 0;
      for (mfem::Array<int> *m : {&currentD_ess, &electPot_ess, &velocity_ess, &pressure_ess})
      {
         m->SetSize(n);
         *m = 0;
      }
   }

   // Mark the listed faces as essential, e.g. SetEssential(velocity_ess, {BoxFace::XMin}).
   static void SetEssential(mfem::Array<int> &marker, std::initializer_list<int> faces)
   {
      for (int f : faces) { marker[f - 1] = 1; }
   }

   // Pointer array in the order LmmhdOperator expects: j, phi, u, p.
   // The pointers point into this object, so call this only once the object
   // is in its final place (i.e. in main, not inside a case function).
   mfem::Array<mfem::Array<int> *> EssentialMarkers()
   {
      mfem::Array<mfem::Array<int> *> m(4);
      m[0] = &currentD_ess;
      m[1] = &electPot_ess;
      m[2] = &velocity_ess;
      m[3] = &pressure_ess;
      return m;
   }
};

inline mfem::real_t pressure_dbc(const mfem::Vector & x)
{
   if (x(0) > 0.0)
   { // Value at outlet.
      return 1.0;
   }
   else
   { // Value at inlet.
      return 1.0;
   }
}

inline void currentD_dbc_vec_func(const mfem::Vector & x, mfem::Vector & f)
{
   f = 0.0;
}

inline mfem::real_t electPot_dbc(const mfem::Vector & x)
{
   return 0.0;
}

inline void zero_func(const mfem::Vector & x, mfem::Vector & f)
{
   f = 0.0;
}

inline mfem::real_t LidRamp(mfem::real_t z, mfem::real_t lid_z, mfem::real_t lid_h)
{
   const mfem::real_t s = (z - (lid_z - lid_h)) / lid_h;   // 0 at z = 1-h, 1 at z = 1
   if (s <= 0.0) { return 0.0; }
   if (s >= 1.0) { return 1.0; }
   return s * s * (3.0 - 2.0 * s);                    // C1.  Smoothly transition from 0 to 1.
}

inline BoundaryConditions LidDrivenCavity(mfem::ParMesh &pmesh, mfem::real_t lid_z)
{
   BoundaryConditions bc(pmesh);
   const mfem::real_t lid_h = 0.054;
   bc.description = "lid-driven cavity, lid at z = " + std::to_string(lid_z);
   bc.velocity_ess = 1;
   bc.velocity = std::make_unique<mfem::VectorFunctionCoefficient>(pmesh.Dimension(),
      [=](const mfem::Vector &x, mfem::Vector &f)
      {
         f = 0.0;
         f(0) = LidRamp(x(2), lid_z, lid_h);
      });
   return bc;
}

/*inline BoundaryConditions SquareDuct(const ParMesh &pmesh)
{
   using namespace BoxFace;
   BoundaryConditions bc(pmesh);
   bc.description = "square duct, inlet x = 0, outlet x = Lx";

   bc.currentD_ess = 1;                                          // insulating walls
   BoundaryConditions::SetEssential(bc.pressure_ess, {XMax});    // outlet
   BoundaryConditions::SetEssential(bc.velocity_ess,
                                    {ZMin, YMin, YMax, XMin, ZMax}); // all but outlet

   bc.velocity = std::make_unique<VectorFunctionCoefficient>(pmesh.Dimension(),
      [](const Vector &x, Vector &f) { f(0) = (9. / 4.) * u_avg * (1. - ((x(1) - y_mid)*(x(1) - y_mid)) / (y_mid * y_mid)) * 
                     (1. - ((x(2) - z_mid)*(x(2) - z_mid)) / (z_mid * z_mid)); });

   return bc;
}*/