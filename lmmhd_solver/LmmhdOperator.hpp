#include "mfem.hpp"
#include "utils.hpp"
#include "VectorConvectionIntegrator.hpp"
#include "constants.hpp"
#include "BlockTriangularPreconditioner.hpp"
#include "CrossProductMatrixCoefficient.hpp"
#include "BlockOperatorPreconditioner.hpp"


using namespace std;
using namespace mfem;


// Operator for solving Ax = b.
class LmmhdOperator : public TimeDependentOperator
{
protected:
   // Finite element spaces
   Array<ParFiniteElementSpace *> spaces;
   // Block offsets for variable access
   Array<int> &block_trueOffsets;

   ParLinearForm *rp, *ru, *rj, *rphi;
   ParBilinearForm *fk, *m, *sp, *mj, *mphi, *dj;
   ParMixedBilinearForm *b, *bT, *g, *gT, *k;
   BlockOperator *A;
   //BlockTriangularPreconditioner *P;
   //BlockDiagonalPreconditioner *P;
   BlockOperatorPreconditioner *P;
   Solver *invF, *invSp, *invMphi, *invM, *invDj, *invGT;

   HypreParMatrix *FkMat = nullptr;
   HypreParMatrix *BMat = nullptr;
   HypreParMatrix *BtMat = nullptr;
   HypreParMatrix *SpMat = nullptr;
   HypreParMatrix *MphiMat = nullptr;
   HypreParMatrix *DjMat = nullptr;

   HypreParMatrix *MjMat = nullptr;
   HypreParMatrix *GMat = nullptr;
   HypreParMatrix *GTMat = nullptr;
   HypreParMatrix *KMat = nullptr;
   HypreParMatrix *KtMat = nullptr;

   // Liquid-metal MHD solver.
   GMRESSolver lmmhd_solver;

   mutable BlockVector *RHS;

   // Boundary conditions.
   Array<Array<int> *> ess_bdr_marker, nat_bdr_marker;

   VectorFunctionCoefficient *currentD_DBC;
   FunctionCoefficient *electPot_DBC;
   VectorFunctionCoefficient *velocity_DBC;
   VectorFunctionCoefficient *magnetic_field_coef;
   FunctionCoefficient *pressure_DBC;

   VectorFunctionCoefficient *zero_coeff;

   // Vectors and coefficients used by linear forms.
   Vector zero_vector;
   Vector one_vector;
   VectorConstantCoefficient one_vector_coef;

   // Need to be kept alive for fk->Assemble() in Solve.
   GridFunction *ustar_n;
   VectorFunctionCoefficient *ucoef;
   VectorGridFunctionCoefficient *ustar_coef = nullptr;

   Vector u_old_true;  // Store previous velocity true DOFs

   real_t massCoefValue;
   ConstantCoefficient *massCoef;

   ConstantCoefficient *zero_coef;

   int dim;
   real_t dt; 

   ParGridFunction *max_flux;

   Vector *magnetic_field;

   CrossProductMatrixCoefficient *C;
   Vector *B;

   

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

