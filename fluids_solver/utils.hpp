#include "mfem.hpp"

using namespace std;
using namespace mfem;


inline void applyDirchValues(const mfem::Vector &k, mfem::Vector &y, mfem::Array<int> dofs)
{
  // FROM SOHAIL'S: https://github.com/sohail69/emfsi_MFEM_app/tree/main
  if(dofs.Size() > 0){ //Only apply if there are constrained DOF's
    const bool use_dev = dofs.UseDevice() || k.UseDevice() || y.UseDevice();
    const int n = dofs.Size();
    // Use read+write access for X - we only modify some of its entries
    auto d_X = y.ReadWrite(use_dev);
    auto d_y = k.Read(use_dev);
    auto d_dofs = dofs.Read(use_dev);
    for (int i = 0; i < n; i++)
    {
      const int dof_i = d_dofs[i];
      if (dof_i >= 0)   d_X[dof_i]    =  d_y[dof_i];
      if (!(dof_i >= 0))d_X[-1-dof_i] = -d_y[-1-dof_i];
    }
  }
};


inline real_t velocity_nbc(const Vector & x)
{
   return 0.0;
}

inline real_t pressure_nbc(const Vector & x)
{
   return 0.0;
}


inline real_t pressure_dbc(const Vector & x)
{
   if (x(0) > 0.0)
   { // Value at outlet.
      return 0.0;
   }
   else
   { // Value at inlet.
      return 1.0;
   }
}

inline real_t zero_dbc(const Vector & x)
{
   return 0.0;
}

inline real_t outflow_term_func(const Vector & x)
{

   real_t val = 1.0;

   if (x(0) < 0.99999)
   { // Value at inlet.
      return 0.0;
   }
   else
   { // Value at outlet.
      return 0.5 * val;
   }
}


inline void velocity_dbc_vec_func(const Vector & x, Vector & f)
{
   //real_t pi = 3.14159;

   // Ensure f is correct size for dimension of problem.
   int dim = x.Size();
   f.SetSize(dim);


   real_t r_max = 0.1;
   real_t u_avg = 1.0;
   
   if (x(0) > 0.00005)
   { // Zero on top and bottom boundaries.
      f(0) = 0.0;
      f(1) = 0.0;

      if (x.Size() == 3)
      {
         f(2) = 0.0;
      }
   } 
   else 
   { // One in x-direction at inlet.
      //f(0) = sin(5 * pi * x(1));
      f(0) = u_avg * (1. - ((x(1)-r_max)*(x(1)-r_max)) / (r_max*r_max));
      //f(0) = 1.0;
      f(1) = 0.0;

      if (x.Size() == 3)
      {
         f(2) = 0.0;
      }
   }
}

inline void u_exact(const mfem::Vector & x, mfem::Vector & f)
{

   real_t r_max = 0.1;
   real_t u_avg = 1.0;

   f(0) = u_avg * (1. - ((x(1)-r_max)*(x(1)-r_max)) / (r_max*r_max));
   f(1) = 0.0;
   if (x.Size() == 3)
   {
      f(2) = 0.0;
   }
   
}


inline void checkpoint(int num)
{
   cout << "**********************************************" << endl;
   cout << "**************** CHECKPOINT " << num << " ****************" << endl;
   cout << "**********************************************" << endl;
   cout << endl;
}

inline void initial_velocity(const Vector & x, Vector & f)
{
   real_t r_max = 0.1;
   real_t u_avg = 1.0;

   f(0) = u_avg * (1. - ((x(1)-r_max)*(x(1)-r_max)) / (r_max*r_max));
}

// Inline visualization
inline void visualize(ParaViewDataCollection &paraview_dc, int order, GridFunction *field, const char *field_name, int ti, double t)
{

   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);

   paraview_dc.SetCycle(ti);
   paraview_dc.SetTime(t);

   // Export field data.
   paraview_dc.RegisterField(field_name,field);

   paraview_dc.Save();
}

inline void CheckConvectionIntegrals(const DenseMatrix &elmat, const DenseMatrix &elmat_conservative, const FiniteElement &el) {
   int num_dofs = el.GetDof();
   
   for (int i = 0; i < num_dofs; i++) {
      for (int j = 0; j < num_dofs; j++) {
         real_t diff = elmat(i, j) + elmat_conservative(i, j); // Should ideally be 0 if they are transposed.
         
         if (std::abs(diff) > 1e-6) {
            std::cerr << "Error: Convection terms do not cancel out. Difference at (" 
                      << i << ", " << j << "): " << diff << std::endl;
         }
      }
   }
}