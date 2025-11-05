// constants.hpp

#include "mfem.hpp"

using namespace std;
using namespace mfem;

// Define constants.
extern real_t Re;
extern real_t Ha;
extern real_t reciprocal_Re;
extern real_t alpha; // alpha = 1 (default).
extern real_t alpha1;
extern real_t neg_alpha1;
extern real_t kappa_val;
extern real_t neg_kappa_val;

extern real_t Bx;
extern real_t By;
extern real_t Bz;

// Define ConstantCoefficients.
extern ConstantCoefficient zero;
extern ConstantCoefficient one;
extern ConstantCoefficient half;
extern ConstantCoefficient neg_one;
extern ConstantCoefficient tmp_const;
extern ConstantCoefficient reciprocal_Re_coeff;
extern ConstantCoefficient alpha1_coeff;
extern ConstantCoefficient neg_alpha1_coeff;

extern ConstantCoefficient kappa;
extern ConstantCoefficient neg_kappa;