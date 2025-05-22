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

   // Block nonlinear form
   //BlockNonlinearForm *Hform;

   // Pressure mass matrix for the preconditioner
   //SparseMatrix *pressure_mass;

   // Newton solver for the hyperelastic operator
   GMRESSolver fluids_solver;

   // Solver for the Jacobian solve in the Newton method
   //Solver *j_solver;

   // Preconditioner
   //Solver *j_prec;

   // Define relevant coefficients
   //Coefficient &mu;

   // Block offsets for variable access
   Array<int> &block_trueOffsets;

   BlockVector *RHS;

public:
   FluidsOperator(Array<FiniteElementSpace *> &fes, Array<Array<int> *>&ess_bdr, Array<Array<int> *> &nat_bdr,
                  Array<int> &block_trueOffsets, int dim);

   // Required to use the native newton solver
   //virtual Operator &GetGradient(const Vector &xp) const;
   virtual void Mult(const Vector &x, Vector &y) const;

   // Driver for the newton solver
   void Solve(Vector &X) const;

   virtual ~FluidsOperator();
};

