// Unit test: the linearised convection operator assembled as in
// LmmhdOperator must be skew-symmetric, C + C^T = 0, which is the property
// that makes O(u*; v, v) = 0 (energy neutrality) hold at the discrete level
// independently of the quadrature rule.
//
// Usage: ./test_convection_skew   (exit code 0 on success)

#include "mfem.hpp"
#include "VectorConvectionIntegrator.hpp"
#include <iostream>

using namespace mfem;

static void ustar_func(const Vector &x, Vector &u)
{
   // An arbitrary, non-solenoidal field that does not vanish on the boundary,
   // so that the test does not rely on div u* = 0 or u*.n = 0.
   u(0) = 1.0 + x(1) * x(2) + 0.3 * x(0);
   u(1) = std::sin(2.0 * x(0)) - 0.5 * x(2);
   u(2) = x(0) * x(1) + 0.2;
}

static real_t SkewDefect(FiniteElementSpace &fes, VectorCoefficient &ustar,
                         real_t a_conservative)
{
   BilinearForm c(&fes);
   c.AddDomainIntegrator(new VectorConvectionIntegrator(ustar, 0.5));
   c.AddDomainIntegrator(new ConservativeVectorConvectionIntegrator(ustar, a_conservative));
   c.Assemble();
   c.Finalize();

   SparseMatrix &C = c.SpMat();
   SparseMatrix *Ct = Transpose(C);
   SparseMatrix *S = Add(1.0, C, 1.0, *Ct);   // C + C^T
   const real_t defect = S->MaxNorm() / C.MaxNorm();
   delete S;
   delete Ct;
   return defect;
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);

   Mesh mesh = Mesh::MakeCartesian3D(3, 3, 3, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
   H1_FECollection fec(2, 3);
   FiniteElementSpace fes(&mesh, &fec, 3);

   VectorFunctionCoefficient ustar_coef(3, ustar_func);
   GridFunction ustar_gf(&fes);
   ustar_gf.ProjectCoefficient(ustar_coef);
   VectorGridFunctionCoefficient ustar(&ustar_gf);

   const real_t defect_fixed = SkewDefect(fes, ustar, 0.5);
   const real_t defect_old = SkewDefect(fes, ustar, -0.5);

   std::cout << "max|C + C^T| / max|C| with a = +0.5 (fixed):    " << defect_fixed << "\n"
             << "max|C + C^T| / max|C| with a = -0.5 (original): " << defect_old << std::endl;

   const bool ok = defect_fixed < 1e-12;
   std::cout << (ok ? "PASS" : "FAIL") << std::endl;
   return ok ? 0 : 1;
}
