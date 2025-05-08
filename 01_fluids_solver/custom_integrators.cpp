// custom_integrators.cpp

#include "custom_integrators.hpp"

VectorConvectionIntegrator::VectorConvectionIntegrator(VectorCoefficient &v, real_t a)
   : velocity(v), alpha(a) { }

void VectorConvectionIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
    int nd = el.GetDof();
    int dim = el.GetDim();
    int vdim = Trans.GetSpaceDim();

    elmat.SetSize(vdim * nd, vdim * nd);
    elmat = 0.0;

    Vector shape(nd);
    shape.SetSize(nd);

    const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), 2 * el.GetOrder());

    Vector vel(dim);
    DenseMatrix dshape(nd, dim);

    for (int i = 0; i < ir->GetNPoints(); i++)
    {
       const IntegrationPoint &ip = ir->IntPoint(i);
       Trans.SetIntPoint(&ip);

       double w = ip.weight * Trans.Weight();
       velocity.Eval(vel, Trans, ip);
       el.CalcPhysDShape(Trans, dshape);
       el.CalcShape(ip, shape);

       for (int j = 0; j < nd; j++)
       {
          double dot = 0.0;
          for (int d = 0; d < dim; d++)
             dot += vel(d) * dshape(j, d);

          for (int k = 0; k < nd; k++)
          {
             for (int vd = 0; vd < vdim; vd++)
             {
                int row = vd * nd + k;
                int col = vd * nd + j;
                elmat(row, col) += alpha * dot * shape(k) * w;
             }
          }
       }
    }
};