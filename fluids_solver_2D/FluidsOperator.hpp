#include "mfem.hpp"
#include "utils.hpp"
#include "custom_integrators.hpp"
#include "constants.hpp"
#include "BlockTriangularPreconditioner.hpp"


using namespace std;
using namespace mfem;


// Operator for solving Ax = b.
class FluidsOperator : public TimeDependentOperator
{
protected:
   // Finite element spaces
   Array<ParFiniteElementSpace *> spaces;
   // Block offsets for variable access
   Array<int> &block_trueOffsets;

   ParLinearForm *rp, *ru;
   ParBilinearForm *fk, *m, *s;
   ParMixedBilinearForm *b, *bT;
   BlockOperator *A;
   //BlockTriangularPreconditioner *P;
   BlockDiagonalPreconditioner *P;
   Solver *invF, *invS;

   HypreParMatrix *FkMat = nullptr;
   HypreParMatrix *BMat = nullptr;
   HypreParMatrix *BtMat = nullptr;
   HypreParMatrix *SMat = nullptr;

   // Fluids solver.
   GMRESSolver fluids_solver;

   mutable BlockVector *RHS;

   // Boundary conditions.
   Array<Array<int> *> ess_bdr_marker, nat_bdr_marker;

   VectorFunctionCoefficient *velocity_DBC;
   VectorFunctionCoefficient *zero_coeff;
   FunctionCoefficient *pressure_DBC;

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

   Vector Xu_dummy, Ru_dummy;

   //ParGridFunction *p_gf;
   

   

public:
   FluidsOperator(Array<ParFiniteElementSpace *> &fes, Array<Array<int> *>&ess_bdr,
                  Array<int> &block_trueOffsets, int dim, double dt);

   void ImplicitSolve(const real_t dt, const Vector &X, Vector &dX_dt);

   void Update(const Vector &X);

   virtual ~FluidsOperator();

   void Set_ustar(GridFunction *u_star)
    {
      if (ustar_coef) { delete ustar_coef; }
      ustar_coef = new VectorGridFunctionCoefficient(u_star);
    }

   void UpdateXwithBCs(BlockVector &X);

};

