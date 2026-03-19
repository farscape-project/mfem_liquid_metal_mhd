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

   GMRESSolver lmmhd_solver;
   //MUMPSSolver lmmhd_solver;
   BlockOperator *A;
   mutable BlockVector *RHS;
   LiPreconditioner *P;

   // Linear and bilinear forms for operator blocks.
   ParLinearForm *rp, *ru, *rj, *rphi;
   ParBilinearForm *fu, *fk, *m, *sp, *mj, *mphi, *dj, *mp;
   ParMixedBilinearForm *b, *bT, *g, *gT, *k;

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
   

public:
   LmmhdOperator(Array<ParFiniteElementSpace *> &fes, Array<Array<int> *>&ess_bdr,
                  Array<int> &block_trueOffsets, int dim, real_t dt);

   void ImplicitSolve(const real_t dt, const Vector &X, Vector &dX_dt);

   void Update(const Vector &X);

   virtual ~LmmhdOperator();

   void Set_ustar(GridFunction *u_star)
    {
      if (ustar_coef) { delete ustar_coef; }
      ustar_coef = new VectorGridFunctionCoefficient(u_star);
    }
};