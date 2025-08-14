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
   ParBilinearForm *fk, *m_p, *s_p;
   ParMixedBilinearForm *b;
   BlockOperator *A;
   //BlockTriangularPreconditioner *P;
   BlockDiagonalPreconditioner *P;

   HypreParMatrix *FkMat = nullptr;
   SparseMatrix *serialFkMat = nullptr;
   SparseMatrix *BMat = nullptr;
   SparseMatrix *BtMat = nullptr;
   SparseMatrix *MpMat = nullptr;
   SparseMatrix *SpMat = nullptr;
   SparseMatrix *LMat = nullptr; 


   // Fluids solver.
   GMRESSolver fluids_solver;

   mutable BlockVector *RHS;

   // Boundary conditions.
   Array<Array<int> *> ess_bdr_marker, nat_bdr_marker;

   VectorFunctionCoefficient *velocity_DBC;
   FunctionCoefficient *pressure_DBC;

   // Vectors and coefficients used by linear forms.
   Vector zero_vector;
   Vector one_vector;
   VectorConstantCoefficient zero_vector_coef;
   VectorConstantCoefficient one_vector_coef;
   FunctionCoefficient velocity_nbc_coeff;
   FunctionCoefficient pressure_nbc_coeff;

   // Need to be kept alive for fk->Assemble() in Solve.
   GridFunction *ustar_n;
   VectorFunctionCoefficient *ucoef;
   VectorGridFunctionCoefficient *ustar_coef = nullptr;

   Vector u_old_true;  // Store previous velocity true DOFs

   double massCoefValue;
   ConstantCoefficient massCoef;

   int dim;
   double dt; 

   ParGridFunction *max_flux;

   

public:
   FluidsOperator(Array<ParFiniteElementSpace *> &fes, Array<Array<int> *>&ess_bdr, Array<Array<int> *> &nat_bdr,
                  Array<int> &block_trueOffsets, int dim, double dt);


   //virtual void Mult(const Vector &x, Vector &y) const;

   //virtual void Mult(const Vector &X, Vector &dX_dt) const;
   //void ImplicitSolve(Vector &X);
   void ImplicitSolve(const real_t dt, const Vector &X, Vector &dX_dt);

   void Update(const Vector &X);

   virtual ~FluidsOperator();

   void Set_ustar(GridFunction *u_star)
    {
      if (ustar_coef) { delete ustar_coef; }
      ustar_coef = new VectorGridFunctionCoefficient(u_star);
    }

   void EnforceDirichletBCs(BlockVector &X);


};

