#include "tools.hpp"

// Print all parameter read in from input file.
void PrintParams(int nx, int ny, int nz, real_t Lx, real_t Ly, real_t Lz,
                 real_t clusterX, real_t clusterY, real_t clusterZ,
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
   std::cout << "clusterX = " << clusterX << std::endl;
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

// Relative change between time levels (global norm of true-dof vectors).
real_t rel_L2_norm(const ParGridFunction &gf, const ParGridFunction &gfN_1, ParFiniteElementSpace *fespace)
{
   ParGridFunction diff_gf(fespace);
   diff_gf = gf;
   diff_gf -= gfN_1;

   Vector diff_t, gf_t;
   diff_gf.GetTrueDofs(diff_t);
   gf.GetTrueDofs(gf_t);

   // Vector::Norml2() is rank-local; ParNormlp reduces over all ranks.
   const real_t l2_diff = ParNormlp(diff_t, 2.0, fespace->GetComm());
   const real_t l2 = ParNormlp(gf_t, 2.0, fespace->GetComm());

   return l2_diff / (l2 + 1e-16);
}


RemoveMeanProjector::RemoveMeanProjector(ParFiniteElementSpace &fes)
   : comm(fes.GetComm())
{
   // Integrals of the basis functions (true-dof vector).
   ParLinearForm mass_lf(&fes);
   mass_lf.AddDomainIntegrator(new DomainLFIntegrator(onecoeff));
   mass_lf.Assemble();

   HypreParVector *tmp = mass_lf.ParallelAssemble();
   mass_vec = *tmp;
   delete tmp;

   // True-dof representation of the constant function 1.
   ParGridFunction one_gf(&fes);
   one_gf.ProjectCoefficient(onecoeff);
   one_gf.GetTrueDofs(one_vec);

   // Global volume.  (The two-argument InnerProduct is rank-local.)
   volume = InnerProduct(comm, mass_vec, one_vec);
}

void RemoveMeanProjector::RemoveMean(Vector &v) const
{
   // Global integral: using the rank-local InnerProduct here subtracted a
   // different constant on every rank.
   const real_t integral = InnerProduct(comm, mass_vec, v);
   const real_t mean = integral / volume;

   v.Add(-mean, one_vec);
}
