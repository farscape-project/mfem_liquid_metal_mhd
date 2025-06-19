#include "mfem.hpp"
#include "utils.hpp"
#include "custom_integrators.hpp"
#include "constants.hpp"


using namespace std;
using namespace mfem;


// Operator for solving Ax = b.
class FluidsOperator : public Operator
{
protected:
   // Finite element spaces
   Array<FiniteElementSpace *> spaces;
   // Block offsets for variable access
   Array<int> &block_trueOffsets;

   mutable GridFunction xu_gf, xp_gf;
   LinearForm *rp, *ru;
   BilinearForm *fk;
   MixedBilinearForm *b;
   BlockOperator *A;

   // Fluids solver.
   GMRESSolver fluids_solver;

   mutable BlockVector *RHS;

   // Boundary conditions.
   Array<Array<int> *> ess_bdr_marker, nat_bdr_marker;

   VectorFunctionCoefficient *velocity_DBC;
   FunctionCoefficient *pressure_DBC;

   int dim;

   // Need to be kept alive for fk->Assemble() in Solve.
   GridFunction ustar_n;
   VectorGridFunctionCoefficient ustar_coef;

   // Vectors and coefficients used by linear forms in FluidsOperator::Solve.
   Vector zero_vector;
   Vector one_vector;
   VectorConstantCoefficient zero_vector_coef;
   VectorConstantCoefficient one_vector_coef;
   FunctionCoefficient velocity_nbc_coeff;
   FunctionCoefficient pressure_nbc_coeff;


public:
   FluidsOperator(Array<FiniteElementSpace *> &fes, Array<Array<int> *>&ess_bdr, Array<Array<int> *> &nat_bdr,
                  Array<int> &block_trueOffsets, int dim);


   virtual void Mult(const Vector &x, Vector &y) const;

   void Solve(Vector &X);

   virtual ~FluidsOperator();
};

