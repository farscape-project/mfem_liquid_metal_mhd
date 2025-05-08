// custom_integrators.cpp

#include "custom_integrators.hpp"

VectorConvectionIntegrator::VectorConvectionIntegrator(VectorCoefficient &v, real_t a)
   : velocity(v), alpha(a) { }

void VectorConvectionIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
    int nd = el.GetDof();               // No. of DOFs in this element.
    int dim = el.GetDim();              // Reference element dimension.
    int vdim = Trans.GetSpaceDim();     // Physical space dimension.

    elmat.SetSize(vdim * nd, vdim * nd);    // Full block matrix.
    elmat = 0.0;

    Vector shape(nd);
    shape.SetSize(nd);

    // Pick quadrature rule.
    const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), 2 * el.GetOrder());

    Vector vel(dim);
    DenseMatrix dshape(nd, dim);

    // Loop through quadrature points and determine element matrix contributions.
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Trans.SetIntPoint(&ip);

        // Calculate Jacobian weight to scale contribution.
        double w = ip.weight * Trans.Weight();
        // Evaluate convection field (Q in hpp file or w in paper).
        velocity.Eval(vel, Trans, ip);
        // Compute derivates of shape functions in physical space.
        el.CalcPhysDShape(Trans, dshape);
        // Evaluate shape function values.
        el.CalcShape(ip, shape);

        // Loop through DOFs.
        for (int j = 0; j < nd; j++)
        {
            double dot = 0.0;
            for (int d = 0; d < dim; d++)
                dot += vel(d) * dshape(j, d); // Calculate dot product Q \cdot \nabla phi_j (basis function).

            for (int k = 0; k < nd; k++)
            {
                for (int vd = 0; vd < vdim; vd++)
                {
                    // Compute integrand for each component (vd - x,y,z).
                    int row = vd * nd + k;
                    int col = vd * nd + j;
                    elmat(row, col) += alpha * dot * shape(k) * w; 
                }
            }
        }
    }
};