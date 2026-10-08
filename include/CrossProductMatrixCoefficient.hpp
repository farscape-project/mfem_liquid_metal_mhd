#pragma once

#include "mfem.hpp"

// Matrix coefficient M such that M v = B x v for a constant vector B.
class CrossProductMatrixCoefficient : public mfem::MatrixCoefficient
{
private:
   mfem::Vector B;

public:
   CrossProductMatrixCoefficient(const mfem::Vector &Bvec)
      : mfem::MatrixCoefficient(Bvec.Size()), B(Bvec) { }

   virtual void Eval(mfem::DenseMatrix &M, mfem::ElementTransformation &T,
                     const mfem::IntegrationPoint &ip) override
   {
      M.SetSize(3);
      mfem::real_t bx = B(0), by = B(1), bz = B(2);

      M(0,0) = 0.0;   M(0,1) = -bz;   M(0,2) =  by;
      M(1,0) =  bz;   M(1,1) = 0.0;   M(1,2) = -bx;
      M(2,0) = -by;   M(2,1) =  bx;   M(2,2) = 0.0;
   }
};
