#pragma once

#include "mfem.hpp"
#include "BoundaryConditions.hpp"
#include "VectorConvectionIntegrator.hpp"
#include "parameters.hpp"
#include "CrossProductMatrixCoefficient.hpp"
#include "LiPreconditioner.hpp"

struct MagneticBlock
{
   // Current operator
   mfem::ParBilinearForm mj;

   // Divergence operators
   mfem::ParMixedBilinearForm g;
   mfem::ParMixedBilinearForm gt;

   // Matrix handles
   mfem::OperatorHandle MjMat_h;
   mfem::OperatorHandle GMat_h;
   mfem::OperatorHandle GtMat_h;

   MagneticBlock(mfem::ParFiniteElementSpace *j_fes, mfem::ParFiniteElementSpace *phi_fes)
        :
         mj(j_fes),
         g(j_fes, phi_fes),
         gt(phi_fes, j_fes)
   {}
};

struct NavierBlock
{
   // Velocity operator
   mfem::ParBilinearForm fu;

   // Divergence operators
   mfem::ParMixedBilinearForm b;
   mfem::ParMixedBilinearForm bt;

   mfem::ParBilinearForm smallPressure;

   // Matrix handles
   mfem::OperatorHandle FuMat_h;
   mfem::OperatorHandle BMat_h;
   mfem::OperatorHandle BtMat_h;
   mfem::OperatorHandle smallPressureMat_h;

   NavierBlock(mfem::ParFiniteElementSpace *u_fes, mfem::ParFiniteElementSpace *p_fes)
         :
         fu(u_fes),
         b(u_fes, p_fes),
         bt(p_fes, u_fes),
         smallPressure(p_fes)
   {}
};

struct CouplingBlock
{
   mfem::ParMixedBilinearForm k;

   // Matrix handles
   mfem::OperatorHandle KMat_h;

   CouplingBlock(mfem::ParFiniteElementSpace *j_fes, mfem::ParFiniteElementSpace *u_fes)
        :
         k(j_fes, u_fes)
   {}
};

struct LiPrecForms
{
   mfem::ParBilinearForm dj, mphi, mp, sp, fk;

   // Matrix handles
   mfem::OperatorHandle DjMat_h;
   mfem::OperatorHandle MphiMat_h;
   mfem::OperatorHandle MpMat_h;
   mfem::OperatorHandle SpMat_h;
   mfem::OperatorHandle FkMat_h;

   LiPrecForms(mfem::ParFiniteElementSpace *j_fes,
               mfem::ParFiniteElementSpace *phi_fes,
               mfem::ParFiniteElementSpace *u_fes,
               mfem::ParFiniteElementSpace *p_fes)
        :
         dj(j_fes),
         mphi(phi_fes),
         mp(p_fes),
         sp(p_fes),
         fk(u_fes)
   {}
};

// Operator for solving Ax = b.
class LmmhdOperator : public mfem::TimeDependentOperator
{
protected:
   lmmhd::Parameters params;

   mfem::real_t theta;
   // Size of the Operator
   int OperatorSize(const mfem::Array<mfem::ParFiniteElementSpace *> spaces);

   // Block problem FE-Spaces and offsets
   mfem::Array<mfem::ParFiniteElementSpace *> spaces;
   mfem::Array<int> &block_trueOffsets;

   //std::unique_ptr<Solver> lmmhd_solver;
   //std::unique_ptr<Operator> A;
   mfem::FGMRESSolver *lmmhd_solver = nullptr;
   mfem::BlockOperator *A = nullptr;

   mfem::BlockVector *X = nullptr, *Xn_1 = nullptr;
   mfem::BlockVector *RHS = nullptr;

   LiPreconditioner *P = nullptr;

   // True once the time-independent blocks and preconditioner pieces exist.
   bool constant_blocks_assembled = false;

   MagneticBlock magnetics;
   NavierBlock fluids;
   CouplingBlock coupling;
   LiPrecForms liprec;

   mfem::ParLinearForm rj, rphi, ru, rp;
   mfem::ParBilinearForm *rhs_mu = nullptr;

   // Vectors and numbers used for bilinear form coefficients.
   mfem::Vector *B = nullptr;
   mfem::Vector *kCoeffVec = nullptr;
   mfem::real_t massCoeffValue;

   // Coefficients used in bilinear forms.
   mfem::ConstantCoefficient *bCoeff = nullptr;
   mfem::ConstantCoefficient *bTCoeff = nullptr;
   mfem::ConstantCoefficient *mjCoeff = nullptr;
   mfem::ConstantCoefficient *gCoeff = nullptr;
   mfem::ConstantCoefficient *gTCoeff = nullptr;
   CrossProductMatrixCoefficient *kCoeff = nullptr;
   mfem::ConstantCoefficient *djCoeff = nullptr;
   mfem::ConstantCoefficient *mphiCoeff = nullptr;
   mfem::ConstantCoefficient *mpCoeff = nullptr;
   mfem::ConstantCoefficient *spCoeff = nullptr;
   mfem::MatrixConstantCoefficient *fkBxVBxVcoeff = nullptr;
   mfem::ConstantCoefficient *fReciprocalReCoeff = nullptr;
   mfem::ConstantCoefficient *alphaCoeff = nullptr;
   mfem::ConstantCoefficient *fMassCoeff = nullptr;
   mfem::ConstantCoefficient *smallPressureCoeff = nullptr;

   mfem::ConstantCoefficient *mpCoeffNorm = nullptr;
   mfem::ConstantCoefficient *spCoeffNorm = nullptr;

   mfem::Array<int> ess_tdof_j, ess_tdof_phi, ess_tdof_u, ess_tdof_p;
   mfem::ParGridFunction *ustar_gf, j_gf, phi_gf, u_gf, p_gf;
   mfem::ParGridFunction ubar_gf;
   mfem::ParGridFunction u_gf_n_1, u_gf_n_2; // Solution history.
   mfem::ParGridFunction j_gf_n_1, phi_gf_n_1, p_gf_n_1; // Solution history.
   std::unique_ptr<mfem::VectorGridFunctionCoefficient> ustar_coef;

   // Discrete divergence RT_k -> L2_k, used to monitor charge conservation.
   std::unique_ptr<mfem::ParDiscreteLinearOperator> div_j;
   mfem::Vector ustar_vec;

   // HypreParMatrices for operator blocks.
   mfem::HypreParMatrix *FuMat = nullptr;
   mfem::HypreParMatrix *FkMat = nullptr;
   mfem::HypreParMatrix *BMat = nullptr;
   mfem::HypreParMatrix *BtMat = nullptr;
   mfem::HypreParMatrix *MpMat = nullptr;
   mfem::HypreParMatrix *SpMat = nullptr;
   mfem::HypreParMatrix *MphiMat = nullptr;
   mfem::HypreParMatrix *DjMat = nullptr;

   mfem::HypreParMatrix *MjMat = nullptr;
   mfem::HypreParMatrix *GMat = nullptr;
   mfem::HypreParMatrix *GtMat = nullptr;
   mfem::HypreParMatrix *KMat = nullptr;
   mfem::HypreParMatrix *KtMat = nullptr;
   mfem::HypreParMatrix *KtFullMat = nullptr; // K^T without essential-dof elimination (for BC on RHS).
   mfem::HypreParMatrix *KMatScaled = nullptr; // -kappa K (momentum-row coupling block).
   mfem::HypreParMatrix *MuMat = nullptr;      // (2/tau) velocity mass matrix for the RHS.
   mfem::HypreParMatrix *BFullMat = nullptr;

   mfem::HypreParMatrix *MpMatNorm = nullptr;
   mfem::HypreParMatrix *SpMatNorm = nullptr;

   // Boundary conditions.
   mfem::Array<mfem::Array<int> *> ess_bdr_marker, nat_bdr_marker;
   mfem::VectorFunctionCoefficient *currentD_DBC = nullptr;
   mfem::FunctionCoefficient *electPot_DBC = nullptr;
   mfem::VectorCoefficient *velocity_DBC = nullptr;
   mfem::VectorFunctionCoefficient *magnetic_field_coef = nullptr;
   mfem::FunctionCoefficient *pressure_DBC = nullptr;
   mfem::VectorGridFunctionCoefficient *velocity_n_1_Coeff = nullptr;
   mfem::VectorSumCoefficient *velocity_bar_DBC = nullptr;

   // Coefficients for zero RHS terms.
   mfem::ConstantCoefficient *zeroCoeff = nullptr;
   mfem::ConstantCoefficient *oneCoeff = nullptr;
   mfem::VectorFunctionCoefficient *vectorZeroCoeff = nullptr;

   int dim;
   Logger &logger;
   FGMRESLogMonitor *fgmres_monitor = nullptr;

   bool remove_pressure_mean, remove_potential_mean;
   RemoveMeanProjector potential_mean_remover;
   RemoveMeanProjector pressure_mean_remover;

   // Outflow BC
   mfem::Array<int> outlet_marker;
   mfem::Array<int> ess_tdof_p_outlet;
   mfem::GridFunctionCoefficient *ustar_x_coeff = nullptr; // x-component of velocity for outflow BC
   mfem::TransformedCoefficient *outflowCoeff = nullptr;
   bool has_outlet = false;

public:
   LmmhdOperator(mfem::Array<mfem::ParFiniteElementSpace *> &fes,
                  BoundaryConditions &bcs,
                  mfem::Array<int> &block_trueOffsets,
                  int dim, 
                  const lmmhd::Parameters &params, 
                  Logger &logger);

   void Step(mfem::real_t &time);

   void UpdateUStar(int step);

   void UpdateHistory();

   mfem::ParGridFunction *GetCurrentDPointer() { return &j_gf; }
   mfem::ParGridFunction *GetPotentialPointer() { return &phi_gf; }
   mfem::ParGridFunction *GetVelocityPointer() { return &u_gf; }
   mfem::ParGridFunction *GetPressurePointer() { return &p_gf; }

   void UpdateIntegrators();

   /// Current-density block solver in the preconditioner: MUMPS (default) or
   /// CG + HypreADS as in Algorithm 4.1.  Call before the first Step().
   //void UseADSForCurrentDensity(bool use_ads) { P->UseADSForCurrentDensity(use_ads); }

   void SetGridFunctionsFromTrueDofs();

   void ReconstructPhysicalVelocityFromUBar(int step);

   mfem::real_t CalcNorms();

   void RemoveMeans();

   void SetBCs();

   void FormASystem();

   void FormPSystem();

   // Build the time-independent blocks of A and of the preconditioner once.
   void SetupConstantBlocks();

   //void CheckMatrices();

   virtual ~LmmhdOperator();


};
