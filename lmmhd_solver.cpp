#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>
#include <cmath>

#include "LmmhdOperator.hpp"
#include "parameters.hpp"
#include "constants.hpp"
#include "InputParser.hpp"
#include "tools.hpp"
#include "scaling_timer.hpp"

using namespace std;
using namespace mfem;


int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE (before anything that queries the rank).
   Mpi::Init(argc, argv);
   Hypre::Init();

   ScalingTimer timer;
   timer.start("Total");
   timer.start("Setup");

   // Only the root rank opens (and truncates) the log file; Logger only
   // writes on the root rank.
   ofstream log_file;
   if (Mpi::Root()) { log_file.open("simulation.log"); }
   Logger logger(log_file);

   mfem::tic();

   // Read in parameters from file.
   InputParser input("params.in");

   lmmhd::Parameters p; // Create parameters object from defaults.

   // Overwrite members of parameters from params.in.
   p.time.dt       = input.GetReal("dt");
   p.time.theta    = input.GetReal("theta");
   p.physics.Re    = input.GetReal("Re");
   p.physics.kappa = input.GetReal("kappa");
   p.physics.B     = {input.GetReal("Bx"), input.GetReal("By"), input.GetReal("Bz")};
   p.physics.alpha = input.GetReal("alpha");

   // Mesh parameters
   int nx = input.GetInt("nx");
   int ny = input.GetInt("ny");
   int nz = input.GetInt("nz");

   real_t Lx = input.GetReal("Lx");
   real_t Ly = input.GetReal("Ly");
   real_t Lz = input.GetReal("Lz");

   real_t clusterX = input.GetReal("clusterX");
   real_t clusterY = input.GetReal("clusterY");
   real_t clusterZ = input.GetReal("clusterZ");

   // Timestepping
   real_t t_final = input.GetReal("t_final");
   int vis_steps = input.GetInt("vis_steps");
   vis_steps = std::max(1, vis_steps);

   std::string save_dir = input.GetString("SaveDirectory");

   real_t Ha = sqrt(p.physics.kappa * p.physics.Re);
   if (Mpi::Root()) { PrintParams(nx, ny, nz, Lx, Ly, Lz, clusterX, clusterY, clusterZ, t_final, p.time.dt, vis_steps, p.physics.Re, p.physics.kappa, Ha); }

   // Generate mesh.
   Mesh mesh = Mesh::MakeCartesian3D(nx, ny, nz, mfem::Element::Type::HEXAHEDRON, Lx, Ly, Lz);
   //Mesh mesh = Mesh::MakeCartesian3D(nx, ny, nz, mfem::Element::Type::TETRAHEDRON, Lx, Ly, Lz);
   //const char *mesh_file = "./mesh/cuboid_clustered.msh";
   //Mesh mesh = Mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   
   // Cluster vertices towards the walls in x, y and z.
   int numVertices = mesh.GetNV();
   for (int i = 0; i < numVertices; i++)
   {
      real_t *v = mesh.GetVertex(i);
      real_t xi = v[0] / Lx; // Normalize x
      real_t yi = v[1] / Ly; // Normalize y
      real_t zi = v[2] / Lz; // Normalize z

      v[0] = cluster_symmetric(xi, clusterX) * Lx;
      v[1] = cluster_symmetric(yi, clusterY) * Ly;
      v[2] = cluster_symmetric(zi, clusterZ) * Lz;
   }

   //ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // Set ramp-up for smooth lid-BC.
   lid_z = Lz;
   //lid_h = Lz * (1.0 - cluster_symmetric((nz - 1.0) / nz, clusterZ));
   //lid_h = 0.027;
   if (Mpi::Root()) { std::cout << "Lid ramp height h = " << lid_h << std::endl; }

   // ----------------------------------------------------------------------------
   // Finite Element Spaces.
   // ----------------------------------------------------------------------------
   // HDiv finite elements for current density.
   RT_FECollection currentD_fec(p.discretisation.order_current_density, dim);
   ParFiniteElementSpace currentD_fespace(&pmesh, &currentD_fec);

   // L2 finite elements for electric scalar potential.
   L2_FECollection electPot_fec(p.discretisation.OrderPotential(), dim);
   ParFiniteElementSpace electPot_fespace(&pmesh, &electPot_fec);

   // H1 continuous Lagrange finite elements (order_pressure + 1)
   // for velocity.
   H1_FECollection velocity_fec(p.discretisation.OrderVelocity(), dim);
   ParFiniteElementSpace velocity_fespace(&pmesh, &velocity_fec, dim);

   // H1 continuous Lagrange finite elements for pressure.
   H1_FECollection pressure_fec(p.discretisation.order_pressure, dim);
   ParFiniteElementSpace pressure_fespace(&pmesh, &pressure_fec);


   Array<ParFiniteElementSpace *> spaces(4);
   spaces[0] = &currentD_fespace;
   spaces[1] = &electPot_fespace;
   spaces[2] = &velocity_fespace;
   spaces[3] = &pressure_fespace;

   int j_space_size = currentD_fespace.GetTrueVSize();
   int phi_space_size = electPot_fespace.GetTrueVSize();
   int v_space_size = velocity_fespace.GetTrueVSize();
   int p_space_size = pressure_fespace.GetTrueVSize();

   // ----------------------------------------------------------------------------
   // Define boundaries.
   // ----------------------------------------------------------------------------
   
   // Essential (Dirichlet) boundary conditions for square duct.
   // Duct flow:           {z0, y0, x1 (outlet), y1, x0 (inlet), z1}
   /*int currentD_bcs[] = {1,  1,  1,           1,  1,          1};
   int electPot_bcs[] = {0,  0,  0,           0,  0,          0};
   int pressure_bcs[] = {0,  0,  1,           0,  0,          0};
   int velocity_bcs[] = {1,  1,  0,           1,  1,          1};*/

   // Essential (Dirichlet) boundary conditions for lid-driven cavity 
   // (lid at z = Lz, see BoundaryConditions.hpp).
   //                     {z0, y0, x1, y1, x0, z1}
   int currentD_bcs[] = {0,  0,  0,  0,  0,  0};
   int electPot_bcs[] = {0,  0,  0,  0,  0,  0};   // L2 space: no boundary dofs
   int pressure_bcs[] = {0,  0,  0,  0,  0,  0};
   int velocity_bcs[] = {1,  1,  1,  1,  1,  1};

   Array<int> ess_boundary_marker_currentD(currentD_bcs, pmesh.bdr_attributes.Max());
   Array<int> ess_boundary_marker_electPot(electPot_bcs, pmesh.bdr_attributes.Max());
   Array<int> ess_boundary_marker_pressure(pressure_bcs, pmesh.bdr_attributes.Max());
   Array<int> ess_boundary_marker_velocity(velocity_bcs, pmesh.bdr_attributes.Max());

   Array<Array<int> *> ess_bdr(4);
   ess_bdr[0] = &ess_boundary_marker_currentD;
   ess_bdr[1] = &ess_boundary_marker_electPot;
   ess_bdr[2] = &ess_boundary_marker_velocity;
   ess_bdr[3] = &ess_boundary_marker_pressure;

   const HYPRE_BigInt j_glob   = currentD_fespace.GlobalTrueVSize();
   const HYPRE_BigInt phi_glob = electPot_fespace.GlobalTrueVSize();
   const HYPRE_BigInt u_glob   = velocity_fespace.GlobalTrueVSize();
   const HYPRE_BigInt p_glob   = pressure_fespace.GlobalTrueVSize();
   const HYPRE_BigInt total_dofs_glob = j_glob + phi_glob + u_glob + p_glob; 

   // Print mesh statistics.
   if (Mpi::Root()) PrintFESpaces(j_glob, phi_glob, u_glob, p_glob);

   // Create save directory if auto option chosen.
   if (save_dir == "auto")
   {
      save_dir = "data_" + std::to_string(total_dofs_glob) + \
         "_kappa_" + std::to_string(std::lround(p.physics.kappa)) + \
         "_Re_" + std::to_string(std::lround(p.physics.Re));
   }

   // Define block structure of the solution vector (j, phi, ubar, p).
   Array<int> block_trueOffsets(5);
   block_trueOffsets[0] = 0;
   block_trueOffsets[1] = j_space_size;
   block_trueOffsets[2] = phi_space_size;
   block_trueOffsets[3] = v_space_size;
   block_trueOffsets[4] = p_space_size;
   block_trueOffsets.PartialSum();

   // Initialise liquid-metal MHD operator.
   LmmhdOperator oper(spaces, ess_bdr, block_trueOffsets, dim, p, logger);

   ParGridFunction *j_gf = oper.GetCurrentDPointer();
   ParGridFunction *phi_gf = oper.GetPotentialPointer();
   ParGridFunction *u_gf = oper.GetVelocityPointer();
   ParGridFunction *p_gf = oper.GetPressurePointer();

   oper.SetGridFunctionsFromTrueDofs();

   // Set up visualisation in Paraview (cycle 0 holds the initial condition).
   ParaViewDataCollection pvdc("lmmhd", &pmesh);
   pvdc.SetPrefixPath(save_dir);
   pvdc.SetDataFormat(VTKFormat::BINARY);
   pvdc.SetHighOrderOutput(false);
   pvdc.SetLevelsOfDetail(2);
   pvdc.SetCycle(0);
   pvdc.SetTime(0.0);
   pvdc.RegisterField("current density", j_gf);
   pvdc.RegisterField("electric potential", phi_gf);
   pvdc.RegisterField("velocity", u_gf);
   pvdc.RegisterField("pressure", p_gf);
   pvdc.Save();

   timer.stop("Setup");

   // Time loop.  An integer step count avoids an extra step from
   // floating-point accumulation of t.
   const int num_steps = static_cast<int>(std::ceil(t_final / p.time.dt - 1e-8));
   real_t t = 0.0;

   for (int ti = 0; ti < num_steps; ti++)
   {
      if (Mpi::Root()) { std::cout << "Time step " << ti << ", time = " << t << ", time elapsed = " << mfem::toc() << std::endl; }
      logger << "Time step " << ti << ", time = " << t << ", time elapsed = " << mfem::toc() << std::endl;

      timer.start("UpdateIntegrators");

      oper.UpdateUStar(ti);     // Extrapolated u* at t^{n+1/2} for the linearised convection term.
      oper.UpdateIntegrators(); // Re-assemble Fu and Fk with the new u*.

      timer.stop("UpdateIntegrators");

      timer.start("Solve");
      oper.Step(t);
      timer.stop("Solve");

      timer.start("UpdatesAfterSolve");
      oper.RemoveMeans();

      oper.SetGridFunctionsFromTrueDofs();

      oper.ReconstructPhysicalVelocityFromUBar(ti);

      const real_t residual = oper.CalcNorms();

      oper.UpdateHistory();
      timer.stop("UpdatesAfterSolve");

      t = (ti + 1) * p.time.dt;

      timer.start("DataIO");
      // Save data.  The velocity is u^{n+1} at time t; the midpoint scheme
      // gives the current density, potential and pressure at t - dt/2.
      if ((ti + 1) % vis_steps == 0 || ti + 1 == num_steps)
      {
         pvdc.SetCycle(ti + 1);
         pvdc.SetTime(t);
         pvdc.Save();
      }
      timer.stop("DataIO");

      if (residual < 1.0e-5)
      {
         if (Mpi::Root()) { std::cout << "Resuidual reached.  Simulation terminated." << std::endl; }
         break;
      }
   }

   if (Mpi::Root()) { std::cout << "Total simulation time: " << mfem::toc() << std::endl; }

   timer.stop("Total");
   timer.report(total_dofs_glob);

   return 0;
}