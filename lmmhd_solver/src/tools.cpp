#include "tools.hpp"

// Print all parameter read in from input file.
void PrintParams(int nx, int ny, int nz, real_t Lx, real_t Ly, real_t Lz,
                 real_t clusterY, real_t clusterZ,
                 real_t t_final, real_t dt, int vis_steps, real_t Re, real_t Ha)
{
   std::cout << " " << std::endl;
   std::cout << "================ Simulation Parameters ================" << std::endl;
   std::cout << "----------- Mesh Parameters -----------" << std::endl;
   std::cout << "nx = " << nx << std::endl;
   std::cout << "ny = " << ny << std::endl;
   std::cout << "nz = " << nz << std::endl;
   std::cout << "Lx = " << Lx << std::endl;
   std::cout << "Ly = " << Ly << std::endl;
   std::cout << "Lz = " << Lz << std::endl;
   std::cout << "clusterY = " << clusterY << std::endl;
   std::cout << "clusterZ = " << clusterZ << std::endl;
   std::cout << " " << std::endl;

   std::cout << "------ Timestepping Parameters --------" << std::endl;
   std::cout << "t_final = " << t_final << std::endl;
   std::cout << "dt = " << dt << std::endl;
   std::cout << "vis_steps = " << vis_steps << std::endl;
   std::cout << " " << std::endl;

   std::cout << "------ Dimensionless Parameters -------" << std::endl;
   std::cout << "Re = " << Re << std::endl;
   std::cout << "Ha = " << Ha << std::endl;
   std::cout << "kappa = " << Ha * Ha / Re << std::endl;
   std::cout << "=======================================================" << std::endl;
   std::cout << " " << std::endl;
   std::cout << " " << std::endl;
}

// Print finite element space sizes.
void PrintFESpaces(int j_space_size, int phi_space_size, int v_space_size, int p_space_size)
{
   std::cout << "=============== Finite Element Spaces =================" << std::endl;
   std::cout << "dim(j) = " << j_space_size << std::endl;
   std::cout << "dim(phi) = " << phi_space_size << std::endl;
   std::cout << "dim(j+phi) = " << j_space_size + phi_space_size << std::endl;
   std::cout << "dim(u) = " << v_space_size << std::endl;
   std::cout << "dim(p) = " << p_space_size << std::endl;
   std::cout << "dim(u+p) = " << v_space_size + p_space_size << std::endl;
   std::cout << "dim(j+phi+u+p) = " << j_space_size + phi_space_size + v_space_size + p_space_size << std::endl;
   std::cout << "=======================================================" << std::endl;
   std::cout << " " << std::endl;
}

// Clustering function (symmetric around center).
real_t cluster_symmetric(real_t xi, real_t factor)
{
    // xi in [0,1], cluster away from 0.5
    real_t x_shifted = xi - 0.5;
    return 0.5 + 0.5 * tanh(factor * x_shifted) / tanh(factor / 2.0);
}

void checkpoint(int num)
{
   cout << "**********************************************" << endl;
   cout << "**************** CHECKPOINT " << num << " ****************" << endl;
   cout << "**********************************************" << endl;
   cout << endl;
}


// Inline visualization
void visualise(ParaViewDataCollection &paraview_dc, int order, GridFunction *field, const char *field_name, int ti, real_t t)
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

// Calculate relative L2-norm of grid functions.
real_t rel_L2_norm(Vector X, Vector Xn_1, ParFiniteElementSpace &fespace)
{
   ParGridFunction gf(&fespace);
   ParGridFunction gfN_1(&fespace);
   ParGridFunction diff_gf(&fespace);
   gf.SetFromTrueDofs(X);
   gfN_1.SetFromTrueDofs(Xn_1);
   diff_gf = gf;
   diff_gf -= gfN_1;

   real_t l2_diff_global = 0.0;
   real_t l2_global = 0.0;

   real_t l2_diff = diff_gf.Norml2();
   real_t l2 = gf.Norml2();

   //real_t rel_l2 = l2_diff / (l2 + 1e-16);

   // Note: work out why MPI_Allreduce is necessary.  I believe the separate threads 
   // should be handled by Norml2() but they don't seem to be.
   //MPI_Allreduce(&l2_diff, &l2_diff_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   //MPI_Allreduce(&l2, &l2_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

   //real_t rel_l2 = l2_diff_global / (l2_global + 1e-16);
   real_t rel_l2 = l2_diff / (l2 + 1e-16);

   return rel_l2;
}