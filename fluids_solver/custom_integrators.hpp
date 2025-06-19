// custom_integrators.hpp

#ifndef VECTOR_CONVECTION_INTEGRATORS_HPP
#define VECTOR_CONVECTION_INTEGRATORS_HPP

#include "mfem.hpp"

using namespace mfem;

// $\alpha (Q \cdot \nabla u, v)$
class VectorConvectionIntegrator : public BilinearFormIntegrator
{
private:
   VectorCoefficient &velocity_coeff;  // Rename to vel_coeff or similar to be clear it's a coefficient.
   real_t alpha;

public:
   VectorConvectionIntegrator(VectorCoefficient &v, real_t a = 1.0);

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat) override;
};


// $-\alpha (v, Q \cdot \nabla v)$, negative transpose of VectorConvectionIntegrator
class ConservativeVectorConvectionIntegrator : public TransposeIntegrator
{
public:
   ConservativeVectorConvectionIntegrator(VectorCoefficient &q, real_t a = 1.0)
      : TransposeIntegrator(new VectorConvectionIntegrator(q, -a)) { }
};

#endif // VECTOR_CONVECTION_INTEGRATORS_HPP




// Experiment to try using pre-exisitng ConvectionIntegrator rather than defining a new one.
/*class VectorConvectionIntegratorCoefficient : public VectorCoefficient
{
private:
   int dim;
   VectorCoefficient &convection_field;

public:
   VectorConvectionIntegratorCoefficient(int dim_, VectorCoefficient &vel_coeff);

   void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip) override;

   virtual ~VectorConvectionIntegratorCoefficient() {}
};*/