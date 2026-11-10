#ifndef VECTOR_CONVECTION_INTEGRATOR_HPP
#define VECTOR_CONVECTION_INTEGRATOR_HPP

#include "mfem.hpp"

// $\alpha (Q \cdot \nabla u, v)$
class VectorConvectionIntegrator : public mfem::BilinearFormIntegrator
{
private:
   mfem::VectorCoefficient &velocity_coeff;
   mfem::real_t alpha;

public:
   VectorConvectionIntegrator(mfem::VectorCoefficient &v, mfem::real_t a = 1.0);

   virtual void AssembleElementMatrix(const mfem::FiniteElement &el,
                                      mfem::ElementTransformation &Trans,
                                      mfem::DenseMatrix &elmat) override;
};


// $-\alpha (v, Q \cdot \nabla v)$, negative transpose of VectorConvectionIntegrator
class ConservativeVectorConvectionIntegrator : public mfem::TransposeIntegrator
{
public:
   ConservativeVectorConvectionIntegrator(mfem::VectorCoefficient &q, mfem::real_t a = 1.0)
      : TransposeIntegrator(new VectorConvectionIntegrator(q, -a)) { }
};

#endif // VECTOR_CONVECTION_INTEGRATOR_HPP