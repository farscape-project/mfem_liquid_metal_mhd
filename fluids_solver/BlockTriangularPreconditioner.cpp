#include "BlockTriangularPreconditioner.hpp"

using namespace mfem;

BlockTriangularPreconditioner::BlockTriangularPreconditioner(const HypreParMatrix &F_,
            const HypreParMatrix &Bt_,
            Array<ParFiniteElementSpace *> &fes,
            Array<int> ess_tdof_p,
            int vsize_, 
            int psize_,
            real_t tau_)
   : Solver(vsize_ + psize_),
      F(F_), Bt(Bt_), 
      x1(vsize_), x2(psize_), y1(vsize_), y2(psize_), tmp(vsize_),
      vsize(vsize_), psize(psize_), tau(tau_)
{
   //fes.Copy(spaces);
   spaces = fes;

   Vector Rp_dummy, Xu_dummy;
   ConstantCoefficient zero_coef(0.0);
   ParLinearForm rp_dummy(spaces[1]);
   rp_dummy.AddDomainIntegrator(new DomainLFIntegrator(zero_coef));
   GridFunction gf(spaces[1]); gf = 0.0;

   m = new ParBilinearForm(spaces[1]);
   m->AddDomainIntegrator(new MassIntegrator());
   m->Assemble(); m->Finalize();
   MMat = new HypreParMatrix();
   m->FormLinearSystem(ess_tdof_p, gf, rp_dummy, *MMat, Xu_dummy, Rp_dummy);

   s = new ParBilinearForm(spaces[1]);
   s->AddDomainIntegrator(new DiffusionIntegrator());
   s->Assemble(); s->Finalize();
   SMat = new HypreParMatrix();
   s->FormLinearSystem(ess_tdof_p, gf, rp_dummy, *SMat, Xu_dummy, Rp_dummy);

   //M_diag = new DSmoother(*MMat);
   M_cg.SetOperator(*MMat);
   M_cg.SetRelTol(1e-5);
   M_cg.SetMaxIter(100);
   M_cg.SetPrintLevel(0);
   //M_cg.SetPreconditioner(*M_diag);  // diagonal preconditioner

   S_amg.SetOperator(*SMat); // stiffness matrix
   S_amg.SetPrintLevel(0);
   S_amg.SetMaxLevels(2); // Approximate 2 iterations

   F_gmres.SetRelTol(1e-3);
   F_gmres.SetMaxIter(200);
   F_gmres.SetPrintLevel(0);
   F_prec = nullptr;
   //F_gmres.SetPreconditioner(*F_prec); // additive Schwarz or other
   
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
   M_cg.Mult(x2, xi);

   // Solve Sp eta = r_p
   Vector eta(psize);
   eta = 0.0;
   S_amg.Mult(x2, eta);

   //std::cout << "tau = " << tau << ", alpha1 = " << alpha1 << std::endl;

   // y_p = -L_p r_p = - alpha1*xi - (2.0/tau)*eta.
   y2 = 0.0;
   add(-alpha1, xi, -1.0/tau, eta, y2);
   //add(-alpha1, xi, -1.0, eta, y2);

   // Simpler pressure preconditioner for now.
   //S_amg.Mult(x2, y2);
   //y2 *= 1.0;

   // Compute tmp = ru - BT y_p
   tmp = 0.0;
   Bt.Mult(y2, tmp);
   tmp.Neg(); 
   tmp += x1;

   // Solve Fk y_u = ru - BT y_p = tmp
   F_gmres.Mult(tmp, y1);

}

void BlockTriangularPreconditioner::SetOperator(const Operator &op) {

   Fmat = static_cast<HypreParMatrix*>(const_cast<Operator*>(&op));
   MFEM_ASSERT(Fmat != nullptr, "Fmat is null!");

   if (F_prec) { delete F_prec; }
   F_prec = new HypreBoomerAMG(F);
   F_prec->SetPrintLevel(0);

   F_gmres.SetPreconditioner(*F_prec);
   F_gmres.SetOperator(F);

   // TODO:
   // Work out why we error when Fmat is used here instead of F.
   // Work out how to read FkMat into this function from FluidsOperator.cpp.

}

BlockTriangularPreconditioner::~BlockTriangularPreconditioner()
{
    //delete M_diag;
    //delete S_gs;
    delete F_prec;
}