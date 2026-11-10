#include "LmmhdTimeStepper.hpp"

LmmhdTimeStepper::LmmhdTimeStepper(mfem::Array<mfem::ParFiniteElementSpace *> &spaces_,
                         BoundaryConditions bcs_,
                         const lmmhd::Parameters &params_,
                         Logger &logger)
   : params(params_),
     bcs(std::move(bcs_)),
     offsets(5)
{
   spaces_.Copy(spaces);

   offsets[0] = 0;
   for (int i = 0; i < 4; i++) { offsets[i + 1] = spaces[i]->GetTrueVSize(); }
   offsets.PartialSum();

   const int dim = spaces[0]->GetParMesh()->Dimension();
   oper = std::make_unique<LmmhdOperator>(spaces, bcs, offsets, dim, params, logger);
   oper->SetGridFunctionsFromTrueDofs();
}

HYPRE_BigInt LmmhdTimeStepper::GlobalTrueVSize() const
{
   HYPRE_BigInt n = 0;
   for (int i = 0; i < 4; i++)
   {
      n += spaces[i]->GlobalTrueVSize();
   }
   return n;
}

mfem::real_t LmmhdTimeStepper::SolveTimeStep()
{
   const mfem::real_t dt = params.time.dt;

   oper->UpdateUStar(ti);
   oper->UpdateIntegrators();

   oper->Step(t);

   oper->RemoveMeans();
   oper->SetGridFunctionsFromTrueDofs();
   oper->ReconstructPhysicalVelocityFromUBar(ti);
   const mfem::real_t residual = oper->CalcNorms();
   oper->UpdateHistory();

   ti++;
   t = ti * dt;

   return residual;
}