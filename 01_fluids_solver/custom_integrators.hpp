// custom_integrators.hpp

#ifndef VECTOR_CONVECTION_INTEGRATORS_HPP
#define VECTOR_CONVECTION_INTEGRATORS_HPP

#include "mfem.hpp"

using namespace mfem;

class VectorConvectionIntegrator : public BilinearFormIntegrator
{
private:
   VectorCoefficient &velocity;
   real_t alpha;

public:
   VectorConvectionIntegrator(VectorCoefficient &v, real_t a = 1.0);

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat) override;
};

// $-\alpha (v, q \cdot \nabla w)$, the negative transpose of VectorConvectionIntegrator
class ConservativeVectorConvectionIntegrator : public TransposeIntegrator
{
public:
   ConservativeVectorConvectionIntegrator(VectorCoefficient &q, real_t a = 1.0)
      : TransposeIntegrator(new VectorConvectionIntegrator(q, -a)) { }
};

#endif // VECTOR_CONVECTION_INTEGRATORS_HPP
