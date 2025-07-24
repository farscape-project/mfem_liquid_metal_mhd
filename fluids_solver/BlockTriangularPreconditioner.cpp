#include "BlockTriangularPreconditioner.hpp"

using namespace mfem;

BlockTriangularPreconditioner::BlockTriangularPreconditioner(const HypreParMatrix &F_,
            const Operator &Bt_,
            ParFiniteElementSpace *pfes_,
            int vsize_, 
            int psize_,
            ConstantCoefficient &vectorMassCoef,
            real_t tau_)
   : Solver(vsize_ + psize_),
      F(F_), Bt(Bt_), pfes(pfes_),
      x1(vsize_), x2(psize_), y1(vsize_), y2(psize_), tmp(vsize_),
      vsize(vsize_), psize(psize_), tau(tau_)
{
   ParBilinearForm *m_p = new ParBilinearForm(pfes);
   m_p->AddDomainIntegrator(new MassIntegrator(one));
   m_p->Assemble(); m_p->Finalize();
   SparseMatrix *MpMat = &(m_p->SpMat());

   ParBilinearForm *s_p = new ParBilinearForm(pfes);
   s_p->AddDomainIntegrator(new DiffusionIntegrator(one));
   s_p->Assemble(); s_p->Finalize();
   //SparseMatrix *SpMat = &(s_p->SpMat());
   HypreParMatrix *SpMat = s_p->ParallelAssemble();

   M_diag = new DSmoother(*MpMat);

   M_cg_solver.SetOperator(*MpMat); // mass matrix
   M_cg_solver.SetRelTol(1e-5);
   M_cg_solver.SetMaxIter(100);
   M_cg_solver.SetPrintLevel(0);
   M_cg_solver.SetPreconditioner(*M_diag);  // diagonal preconditioner

   S_amg.SetOperator(*SpMat); // stiffness matrix
   S_amg.SetPrintLevel(0);
   S_amg.SetMaxLevels(2); // Approximate 2 iterations

   // Not currently preconditioned.
   F_gmres.SetRelTol(1e-3);
   F_gmres.SetMaxIter(200);
   F_gmres.SetPrintLevel(0);
   //F_gmres.SetPreconditioner(*F_prec); // additive Schwarz or other
   
}

void BlockTriangularPreconditioner::Update(const HypreParMatrix &F)
{
   //F_prec = Hypre_ParCSR(F);
   //F_gmres.SetPreconditioner(*F_prec);
   F_gmres.SetOperator(F);
}

void BlockTriangularPreconditioner::Mult(const Vector &x, Vector &y) const
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

   //std::cout << "tau = " << tau << ", alpha1 = " << alpha1 << std::endl;

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

void BlockTriangularPreconditioner::SetOperator(const Operator &op) {}

BlockTriangularPreconditioner::~BlockTriangularPreconditioner()
{
    delete M_diag;
    delete S_gs;
}