#include "mfem.hpp"
#include "constants.hpp"

using namespace mfem;

// This preconditioner solves y = Px such that
// P = [ F_k  B^T  ]
//     [ 0    L    ]
// where F_k is the velocity mass matrix, B^T is the transpose of the divergence operator,
// and L contains a mass matrix and stiffness matrix for the pressure.

class BlockTriangularPreconditioner : public Solver
{
private:
   const Operator &F, &Bt;
   FiniteElementSpace *pfes;
   Solver *L_solver;
   CGSolver M_cg_solver;
   CGSolver S_amg;
   //GeometricMultigrid S_amg;
   GMRESSolver F_gmres;

   mutable Vector x1, x2, y1, y2, tmp;

   int vsize, psize;

   DSmoother *M_diag;
   GSSmoother *S_gs;

   real_t tau;

public:
   BlockTriangularPreconditioner(const Operator &F_,
                  const Operator &Bt_,
                  FiniteElementSpace *pfes_,
                  int vsize_, 
                  int psize_,
                  ConstantCoefficient &vectorMassCoef,
                  real_t tau_)
      : Solver(vsize_ + psize_),
        F(F_), Bt(Bt_), pfes(pfes_),
        x1(vsize_), x2(psize_), y1(vsize_), y2(psize_), tmp(vsize_),
        vsize(vsize_), psize(psize_), tau(tau_)
   {
      BilinearForm *m_p = new BilinearForm(pfes);
      m_p->AddDomainIntegrator(new MassIntegrator(one));
      m_p->Assemble(); m_p->Finalize();
      SparseMatrix *MpMat = &(m_p->SpMat());

      BilinearForm *s_p = new BilinearForm(pfes);
      s_p->AddDomainIntegrator(new DiffusionIntegrator(one));
      s_p->Assemble(); s_p->Finalize();
      SparseMatrix *SpMat = &(s_p->SpMat());

      M_diag = new DSmoother(*MpMat);

      M_cg_solver.SetOperator(*MpMat); // mass matrix
      M_cg_solver.SetRelTol(1e-5);
      M_cg_solver.SetMaxIter(100);
      M_cg_solver.SetPrintLevel(0);
      M_cg_solver.SetPreconditioner(*M_diag);  // diagonal preconditioner

      S_gs = new GSSmoother(*SpMat);

      // Next step: HypreBoomerAMG solver on Glados.
      S_amg.SetOperator(*SpMat); // stiffness matrix
      S_amg.SetPrintLevel(0);
      S_amg.SetMaxIter(100); // Approximate 2 iterations
      S_amg.SetRelTol(1e-6);
      S_amg.SetPreconditioner(*S_gs);  // diagonal preconditioner

      F_gmres.SetOperator(F);
      F_gmres.SetRelTol(1e-3);
      F_gmres.SetMaxIter(200);
      F_gmres.SetPrintLevel(0);
      //F_gmres.SetPreconditioner(*F_prec); // additive Schwarz or other
      
   }

   virtual void Mult(const Vector &x, Vector &y) const override
   {

      x1.MakeRef(const_cast<Vector &>(x), 0, vsize);
      x2.MakeRef(const_cast<Vector &>(x), vsize, psize);

      y1.MakeRef(y, 0, vsize);
      y2.MakeRef(y, vsize, psize);

      // Solve Mp xi = r_p
      Vector xi(psize);
      xi = 0.0;
      M_cg_solver.Mult(x2, xi);

      // Solve Sp eta = r_p
      Vector eta(psize);
      eta = 0.0;
      S_amg.Mult(x2, eta);

      //std::cout << "alpha1 = " << alpha1 << std::endl;

      // y_p = -L_p r_p = - alpha1*xi - eta.
      y2 = 0.0;
      add(-alpha1, xi, -2.0/tau, eta, y2);
      //add(-alpha1, xi, -1.0, eta, y2);

      // Compute tmp = ru - BT y_p
      tmp = 0.0;
      Bt.Mult(y2, tmp);
      tmp.Neg(); 
      tmp += x1;

      // Solve Fk y_u = ru - BT y_p = tmp
      F_gmres.Mult(tmp, y1);

   }

      virtual void SetOperator(const Operator &op) override {}

};




/*virtual void Mult(const Vector &x, Vector &y) const override
   {
      // This preconditioner solves for y = [y1, y2] such that
      // F_k y1 + B^T y2 = x1,   (1)
      // L y2 = x2,              (2)
      // where x = [x1, x2] is the input vector.  First, equation 2 is solved by
      // calculating y2 = L^{-1} x2. Then, equation 1 is solved by calculating
      // y1 = F_k^{-1} (x1 - B^T y2).

      x1.MakeRef(const_cast<Vector &>(x), 0, vsize);
      x2.MakeRef(const_cast<Vector &>(x), vsize, psize);

      y1.MakeRef(y, 0, vsize);
      y2.MakeRef(y, vsize, psize);

      // Step 1: y2 = L^{-1} x2
      L_solver->Mult(x2, y2);

      // Step 2: tmp = x1 - BT * y2
      Bt.Mult(y2, tmp);
      tmp.Neg(); 
      tmp += x1;

      // Step 3: y1 = F^{-1} tmp
      F.Mult(tmp, y1);
   }*/