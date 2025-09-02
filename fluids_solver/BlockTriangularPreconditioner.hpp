#include "mfem.hpp"
#include "constants.hpp"
#include "linalg/hypre.hpp"

using namespace mfem;

// This preconditioner solves y = Px such that
// P = [ F_k  B^T  ]
//     [ 0    L    ]
// where F_k is the velocity mass matrix, B^T is the transpose of the divergence operator,
// and L contains a mass matrix and stiffness matrix for the pressure.

class BlockTriangularPreconditioner : public Solver
{
private:
   const HypreParMatrix &F;
   const Operator &Bt;
   Array<ParFiniteElementSpace *> spaces;
   CGSolver M_cg;
   HypreBoomerAMG *F_prec;
   HypreBoomerAMG S_amg;
   GMRESSolver F_gmres;

   mutable Vector x1, x2, y1, y2, tmp;

   int vsize, psize;

   //DSmoother *M_diag;
   //GSSmoother *S_gs;

   real_t tau;


   ParBilinearForm *m, *s;
   HypreParMatrix *MMat, *SMat, *Fmat;


   //Hypre_ParCSR *F_prec;

public:
   BlockTriangularPreconditioner(const HypreParMatrix &F_,
                  const HypreParMatrix &Bt_,
                  Array<ParFiniteElementSpace *> &fes,
                  Array<int> ess_tdof_p,
                  int vsize_, 
                  int psize_,
                  real_t tau_);

   virtual void Mult(const Vector &x, Vector &y) const override;
   virtual void SetOperator(const Operator &op) override;

   virtual ~BlockTriangularPreconditioner();
};