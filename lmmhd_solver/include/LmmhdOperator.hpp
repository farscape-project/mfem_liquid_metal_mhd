#include "mfem.hpp"
#include "BoundaryConditions.hpp"
#include "VectorConvectionIntegrator.hpp"
#include "constants.hpp"
#include "CrossProductMatrixCoefficient.hpp"
#include "LiPreconditioner.hpp"


using namespace std;
using namespace mfem;


// Operator for solving Ax = b.
class LmmhdOperator : public TimeDependentOperator
{
protected:

   Array<ParFiniteElementSpace *> spaces;
   Array<int> &block_trueOffsets;

   FGMRESSolver *lmmhd_solver;
   BlockOperator *A;
   mutable BlockVector *RHS;
   LiPreconditioner *P;

   OrthoSolver *prec_ortho_solver;

   // Linear and bilinear forms for operator blocks.
   ParLinearForm *rp, *ru, *rj, *rphi;
   ParBilinearForm *fu, *fk, *m, *sp, *mj, *mphi, *dj, *mp;
   ParMixedBilinearForm *b, *bT, *g, *gT, *k;

   ParBilinearForm *mpNorm, *spNorm;

   // Vectors and numbers used for bilinear form coefficients.
   Vector *B;
   Vector *kCoeffVec;
   real_t massCoeffValue;

   // Coefficients used in bilinear forms.
   ConstantCoefficient *bCoeff;
   ConstantCoefficient *bTCoeff;
   ConstantCoefficient *mjCoeff;
   ConstantCoefficient *gCoeff;
   ConstantCoefficient *gTCoeff;
   CrossProductMatrixCoefficient *kCoeff;
   ConstantCoefficient *djCoeff;
   ConstantCoefficient *mphiCoeff;
   ConstantCoefficient *mpCoeff;
   ConstantCoefficient *spCoeff;
   MatrixConstantCoefficient *fkBxVBxVcoeff;
   ConstantCoefficient *fReciprocalReCoeff;
   ConstantCoefficient *fMassCoeff;

   ConstantCoefficient *mpCoeffNorm;
   ConstantCoefficient *spCoeffNorm;

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
   HypreParMatrix *GTMat = nullptr;
   HypreParMatrix *KMat = nullptr;
   HypreParMatrix *KtMat = nullptr;

   HypreParMatrix *MpMatNorm = nullptr;
   HypreParMatrix *SpMatNorm = nullptr;

   // Boundary conditions.
   Array<Array<int> *> ess_bdr_marker, nat_bdr_marker;
   VectorFunctionCoefficient *currentD_DBC;
   FunctionCoefficient *electPot_DBC;
   VectorFunctionCoefficient *velocity_DBC;
   VectorFunctionCoefficient *magnetic_field_coef;
   FunctionCoefficient *pressure_DBC;
   
   // Coefficients for zero RHS terms.
   ConstantCoefficient *zeroCoeff;
   VectorFunctionCoefficient *vectorZeroCoeff;

   // Need to be kept alive for fk->Assemble() in Solve.
   GridFunction *ustar_n;
   VectorFunctionCoefficient *ucoef;
   VectorGridFunctionCoefficient *ustar_coef = nullptr;

   int dim;
   real_t dt;
   int debug;

   // Required for MeanZero function.
   ParLinearForm *mass_lf = nullptr;
   ConstantCoefficient onecoeff;
   real_t volume = 0.0;
   bool numerical_integ = false;
   IntegrationRules gll_rules;

public:
   LmmhdOperator(Array<ParFiniteElementSpace *> &fes, Array<Array<int> *>&ess_bdr,
                  Array<int> &block_trueOffsets, int dim, real_t dt, int debug);

   void ImplicitSolve(const real_t dt, const Vector &X, Vector &dX_dt);

   void Update(const Vector &X);

   virtual ~LmmhdOperator();

   void Set_ustar(GridFunction *u_star)
    {
      if (ustar_coef) { delete ustar_coef; }
      ustar_coef = new VectorGridFunctionCoefficient(u_star);
    }

    void MeanZero(ParGridFunction &v)
   {
      // Currently just copied from navier_solver.cpp with some tweaks to get working.
      // Make sure not to recompute the inner product linear form every
      // application.
      int order = 1;  // Temporarily hardcoding order.

      if (mass_lf == nullptr)
      {
         onecoeff.constant = 1.0;
         mass_lf = new ParLinearForm(v.ParFESpace());
         auto *dlfi = new DomainLFIntegrator(onecoeff);
         if (numerical_integ)
         {
            const IntegrationRule &ir_ni = gll_rules.Get(spaces[3]->GetFE(0)->GetGeomType(),
                                                         2 * order - 1);
            dlfi->SetIntRule(&ir_ni);
         }
         mass_lf->AddDomainIntegrator(dlfi);
         mass_lf->Assemble();

         ParGridFunction one_gf(v.ParFESpace());
         one_gf.ProjectCoefficient(onecoeff);

         volume = mass_lf->operator()(one_gf);
      }

      real_t integ = mass_lf->operator()(v);

      v -= integ / volume;
   }
};