#pragma once
#include "mfem.hpp"
#include "parameters.hpp"
#include "BoundaryConditions.hpp"
#include "LmmhdOperator.hpp"

class LmmhdTimeStepper
{
public:
   LmmhdTimeStepper(mfem::Array<mfem::ParFiniteElementSpace *> &spaces,
               BoundaryConditions bcs,
               const lmmhd::Parameters &params,
               Logger &logger);

   LmmhdTimeStepper(const LmmhdTimeStepper &) = delete;
   LmmhdTimeStepper &operator=(const LmmhdTimeStepper &) = delete;

   mfem::ParGridFunction &CurrentDensity() { return *oper->GetCurrentDPointer(); }
   mfem::ParGridFunction &Potential()      { return *oper->GetPotentialPointer(); }
   mfem::ParGridFunction &Velocity()       { return *oper->GetVelocityPointer(); }
   mfem::ParGridFunction &Pressure()       { return *oper->GetPressurePointer(); }

   HYPRE_BigInt GlobalTrueVSize() const;

   // Take one complete time step.  Returns the residual from CalcNorms().
   mfem::real_t SolveTimeStep();
   mfem::real_t Time() const { return t; }
   int StepIndex() const { return ti; }

private:
   lmmhd::Parameters params;
   BoundaryConditions bcs;
   mfem::Array<mfem::ParFiniteElementSpace *> spaces;
   mfem::Array<int> offsets;
   std::unique_ptr<LmmhdOperator> oper;
   mfem::real_t t = 0.0;
   int ti = 0;
};
