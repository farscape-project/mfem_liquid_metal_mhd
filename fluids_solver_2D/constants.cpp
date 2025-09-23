// constants.hpp

#include "mfem.hpp"
#include "constants.hpp"

using namespace std;
using namespace mfem;

// Define constants.
real_t Re(1.0);
real_t reciprocal_Re(1 / Re);
real_t alpha(1.0); // alpha = 1 (default).
real_t alpha1(alpha + reciprocal_Re);
real_t neg_alpha1(-alpha1);
//real_t tau(1e-6);

// Define ConstantCoefficients.
ConstantCoefficient zero(0.0);
ConstantCoefficient one(1.0);
ConstantCoefficient half(0.5);
ConstantCoefficient neg_one(-1.0);
ConstantCoefficient tmp_const(1.0);
ConstantCoefficient reciprocal_Re_coef(reciprocal_Re);
ConstantCoefficient alpha1_coeff(alpha1);
ConstantCoefficient neg_alpha1_coeff(neg_alpha1);

