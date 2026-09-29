#include "mfem.hpp"
#include "BoundaryConditions.hpp"
#include "VectorConvectionIntegrator.hpp"
#include "constants.hpp"
#include "CrossProductMatrixCoefficient.hpp"
#include "LiPreconditioner.hpp"

using namespace std;
using namespace mfem;

struct MagneticBlock
{
   // Current operator
   ParBilinearForm mj;

   // Divergence operators
   ParMixedBilinearForm g;
   ParMixedBilinearForm gt;

   // Matrix handles
   OperatorHandle MjMat_h;
   OperatorHandle GMat_h;
   OperatorHandle GtMat_h;

   MagneticBlock(ParFiniteElementSpace *j_fes, ParFiniteElementSpace *phi_fes)
        :
         mj(j_fes),
         g(j_fes, phi_fes),
         gt(phi_fes, j_fes)
   {}
};

struct NavierBlock
{
   // Velocity operator
   ParBilinearForm fu;

   // Divergence operators
   ParMixedBilinearForm b;
   ParMixedBilinearForm bt;

   ParBilinearForm smallPressure;

   // Matrix handles
   OperatorHandle FuMat_h;
   OperatorHandle BMat_h;
   OperatorHandle BtMat_h;
   OperatorHandle smallPressureMat_h;

   NavierBlock(ParFiniteElementSpace *u_fes, ParFiniteElementSpace *p_fes)
         :
         fu(u_fes),
         b(u_fes, p_fes),
         bt(p_fes, u_fes),
         smallPressure(p_fes)
   {}
};

struct CouplingBlock
{
   ParMixedBilinearForm k;

   // Matrix handles
   OperatorHandle KMat_h;

   CouplingBlock(ParFiniteElementSpace *j_fes, ParFiniteElementSpace *u_fes)
        :
         k(j_fes, u_fes)
   {}
};

struct LiPrecForms
{
   ParBilinearForm dj, mphi, mp, sp, fk;

   // Matrix handles
   OperatorHandle DjMat_h;
   OperatorHandle MphiMat_h;
   OperatorHandle MpMat_h;
   OperatorHandle SpMat_h;
   OperatorHandle FkMat_h;

   LiPrecForms(ParFiniteElementSpace *j_fes,
               ParFiniteElementSpace *phi_fes,
               ParFiniteElementSpace *u_fes,
               ParFiniteElementSpace *p_fes)
        :
         dj(j_fes),
         mphi(phi_fes),
         mp(p_fes),
         sp(p_fes),
         fk(u_fes)
   {}
};

// Operator for solving Ax = b.
class LmmhdOperator : public TimeDependentOperator
{
protected:
   // Size of the Operator
   int OperatorSize(const Array<ParFiniteElementSpace *> spaces);

   // Block problem FE-Spaces and offsets
   Array<ParFiniteElementSpace *> spaces;
   Array<int> &block_trueOffsets;

   //std::unique_ptr<Solver> lmmhd_solver;
   //std::unique_ptr<Operator> A;
   FGMRESSolver *lmmhd_solver = nullptr;
   BlockOperator *A = nullptr;

   BlockVector *X = nullptr, *Xn_1 = nullptr;
   BlockVector *RHS = nullptr;

   LiPreconditioner *P = nullptr;

   // True once the time-independent blocks and preconditioner pieces exist.
   bool constant_blocks_ready = false;

   MagneticBlock magnetics;
   NavierBlock fluids;
   CouplingBlock coupling;
   LiPrecForms liprec;

   ParLinearForm rj, rphi, ru, rp;
   ParBilinearForm *rhs_mu = nullptr;

   // Vectors and numbers used for bilinear form coefficients.
   Vector *B = nullptr;
   Vector *kCoeffVec = nullptr;
   real_t massCoeffValue;

   // Coefficients used in bilinear forms.
   ConstantCoefficient *bCoeff = nullptr;
   ConstantCoefficient *bTCoeff = nullptr;
   ConstantCoefficient *mjCoeff = nullptr;
   ConstantCoefficient *gCoeff = nullptr;
   ConstantCoefficient *gTCoeff = nullptr;
   CrossProductMatrixCoefficient *kCoeff = nullptr;
   ConstantCoefficient *djCoeff = nullptr;
   ConstantCoefficient *mphiCoeff = nullptr;
   ConstantCoefficient *mpCoeff = nullptr;
   ConstantCoefficient *spCoeff = nullptr;
   MatrixConstantCoefficient *fkBxVBxVcoeff = nullptr;
   ConstantCoefficient *fReciprocalReCoeff = nullptr;
   ConstantCoefficient *alphaCoeff = nullptr;
   ConstantCoefficient *fMassCoeff = nullptr;
   ConstantCoefficient *smallPressureCoeff = nullptr;

   ConstantCoefficient *mpCoeffNorm = nullptr;
   ConstantCoefficient *spCoeffNorm = nullptr;

   Array<int> ess_tdof_j, ess_tdof_phi, ess_tdof_u, ess_tdof_p;
   ParGridFunction *ustar_gf, j_gf, phi_gf, u_gf, p_gf;
   ParGridFunction ubar_gf;
   ParGridFunction u_gf_n_1, u_gf_n_2; // Solution history.
   ParGridFunction j_gf_n_1, phi_gf_n_1, p_gf_n_1; // Solution history.
   std::unique_ptr<VectorGridFunctionCoefficient> ustar_coef;

   // Discrete divergence RT_k -> L2_k, used to monitor charge conservation.
   std::unique_ptr<ParDiscreteLinearOperator> div_j;
   Vector ustar_vec;

   // HypreParMatrices for operator blocks.
   HypreParMatrix *FuMat = nullptr;
   HypreParMatrix *FkMat = nullptr;
   HypreParMatrix *BMat = nullptr;
   HypreParMatrix *BtMat = nullptr;
   HypreParMatrix *MpMat = nullptr;
   HypreParMatrix *SpMat = nullptr;
   HypreParMatrix *MphiMat = nullptr;
   HypreParMatrix *DjMat = nullptr;

   HypreParMatrix *MjMat = nullptr;
   HypreParMatrix *GMat = nullptr;
   HypreParMatrix *GtMat = nullptr;
   HypreParMatrix *KMat = nullptr;
   HypreParMatrix *KtMat = nullptr;
   HypreParMatrix *KtFullMat = nullptr; // K^T without essential-dof elimination (for BC lifting).
   HypreParMatrix *KMatScaled = nullptr; // -kappa K (momentum-row coupling block).
   HypreParMatrix *MuMat = nullptr;      // (2/tau) velocity mass matrix for the RHS.

   HypreParMatrix *MpMatNorm = nullptr;
   HypreParMatrix *SpMatNorm = nullptr;

   // Boundary conditions.
   Array<Array<int> *> ess_bdr_marker, nat_bdr_marker;
   VectorFunctionCoefficient *currentD_DBC = nullptr;
   FunctionCoefficient *electPot_DBC = nullptr;
   VectorFunctionCoefficient *velocity_DBC = nullptr;
   VectorFunctionCoefficient *magnetic_field_coef = nullptr;
   FunctionCoefficient *pressure_DBC = nullptr;
   VectorGridFunctionCoefficient *velocity_n_1_Coeff = nullptr;
   VectorSumCoefficient *velocity_bar_DBC = nullptr;

   // Coefficients for zero RHS terms.
   ConstantCoefficient *zeroCoeff = nullptr;
   ConstantCoefficient *oneCoeff = nullptr;
   VectorFunctionCoefficient *vectorZeroCoeff = nullptr;

   int dim;
   real_t dt;
   int debug;
   Logger &logger;
   FGMRESLogMonitor *fgmres_monitor = nullptr;

   RemoveMeanProjector potential_mean_remover;
   RemoveMeanProjector pressure_mean_remover;

public:
   LmmhdOperator(Array<ParFiniteElementSpace *> &fes, Array<Array<int> *>&ess_bdr,
                  Array<int> &block_trueOffsets, int dim, real_t dt, int debug, 
                  Logger &logger);

   void Step(real_t &time, real_t dt);

   void UpdateUStar(int step);

   void UpdateHistory();

   ParGridFunction *GetCurrentDPointer() { return &j_gf; }
   ParGridFunction *GetPotentialPointer() { return &phi_gf; }
   ParGridFunction *GetVelocityPointer() { return &u_gf; }
   ParGridFunction *GetPressurePointer() { return &p_gf; }

   void UpdateIntegrators();

   /// Current-density block solver in the preconditioner: MUMPS (default) or
   /// CG + HypreADS as in Algorithm 4.1.  Call before the first Step().
   void UseADSForCurrentDensity(bool use_ads) { P->UseADSForCurrentDensity(use_ads); }

   void SetGridFunctionsFromTrueDofs();

   void ReconstructPhysicalVelocityFromUBar(int step);

   void CalcNorms();

   void RemoveMeans();

   void SetBCs();

   void FormASystem();

   void FormPSystem();

   // Build the time-independent blocks of A and of the preconditioner once.
   void SetupConstantBlocks();

   //void CheckMatrices();

   virtual ~LmmhdOperator();


};
