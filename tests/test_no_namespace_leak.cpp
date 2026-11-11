// Compile-only check: library headers must not leak 'using namespace' into
// files that include them (as MOOSE/libMesh files will).
// Built by 'make check-headers'; never linked or run.

// Names that exist in MFEM / std and also in libMesh or MOOSE.
struct Vector {};
struct Mesh {};
struct Point {};
template <typename T> struct Array {};
struct string {};

#include "parameters.hpp"
#include "BoundaryConditions.hpp"
#include "Discretisation.hpp"
#include "LmmhdTimeStepper.hpp"

int main()
{
   // If any header above did 'using namespace mfem;' or 'using namespace std;',
   // these become ambiguous and compilation fails.
   Vector v;
   Mesh m;
   Point p;
   Array<int> a;
   string s;
   (void)v; (void)m; (void)p; (void)a; (void)s;
   return 0;
}
