#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>
#include <cmath>

#include "LmmhdTimeStepper.hpp"
#include "LmmhdOperator.hpp"
#include "parameters.hpp"
#include "Discretisation.hpp"
#include "InputParser.hpp"
#include "tools.hpp"
#include "scaling_timer.hpp"

using namespace std;
using namespace mfem;

// Select case, e.g. lid-driven cavity or square duct.
static BoundaryConditions MakeBoundaryConditions(const std::string &name, ParMesh &pmesh, const InputParser &input)
{
   if (name == "lid_driven_cavity") { return LidDrivenCavity(pmesh, input.GetReal("Lz")); }
   //if (name == "square_duct") { return SquareDuct(pmesh); }
   MFEM_ABORT("Unknown Case '" << name << "' in params.in (lid_driven_cavity)");// | square_duct)");
   return BoundaryConditions(pmesh);
}

int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   Hypre::Init();

   ScalingTimer timer;
   timer.start("Total");
   timer.start("Setup");

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

   mesh.Save("box_clustered.mesh"); // Save mesh to file.

   //ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // Finite Element Spaces.
   Discretisation disc(pmesh, p.discretisation);
   Array<ParFiniteElementSpace *> &spaces = disc.Spaces();

   int j_space_size = spaces[0]->GetTrueVSize();
   int phi_space_size = spaces[1]->GetTrueVSize();
   int v_space_size = spaces[2]->GetTrueVSize();
   int p_space_size = spaces[3]->GetTrueVSize();

   // Define boundary conditions.
   BoundaryConditions bcs = MakeBoundaryConditions(input.GetString("Case"), pmesh, input);
   //if (Mpi::Root()) { std::cout << "Boundary conditions: " << bcs.description << std::endl; }
   Array<Array<int> *> ess_bdr = bcs.EssentialMarkers();

   // Print mesh statistics.
   const HYPRE_BigInt j_glob   = spaces[0]->GlobalTrueVSize();
   const HYPRE_BigInt phi_glob = spaces[1]->GlobalTrueVSize();
   const HYPRE_BigInt u_glob   = spaces[2]->GlobalTrueVSize();
   const HYPRE_BigInt p_glob   = spaces[3]->GlobalTrueVSize();
   const HYPRE_BigInt total_dofs_glob = j_glob + phi_glob + u_glob + p_glob; 
   if (Mpi::Root()) PrintFESpaces(j_glob, phi_glob, u_glob, p_glob);

   // Create save directory if auto option chosen.
   if (save_dir == "auto")
   {
      save_dir = "data_" + std::to_string(total_dofs_glob) + \
         "_kappa_" + std::to_string(std::lround(p.physics.kappa)) + \
         "_Re_" + std::to_string(std::lround(p.physics.Re));
   }

   LmmhdTimeStepper timeStepper(spaces, MakeBoundaryConditions(input.GetString("Case"), pmesh, input), p, logger);

   // Set up visualisation in Paraview (cycle 0 holds the initial condition).
   ParaViewDataCollection pvdc("lmmhd", &pmesh);
   pvdc.SetPrefixPath(save_dir);
   pvdc.SetDataFormat(VTKFormat::BINARY);
   pvdc.SetHighOrderOutput(false);
   pvdc.SetLevelsOfDetail(2);
   pvdc.SetCycle(0);
   pvdc.SetTime(0.0);
   pvdc.RegisterField("current density", &timeStepper.CurrentDensity());
   pvdc.RegisterField("electric potential", &timeStepper.Potential());
   pvdc.RegisterField("velocity", &timeStepper.Velocity());
   pvdc.RegisterField("pressure", &timeStepper.Pressure());
   pvdc.Save();

   timer.stop("Setup");

   // Time loop.
   const int num_steps = static_cast<int>(std::ceil(t_final / p.time.dt - 1e-8));

   while (timeStepper.StepIndex() < num_steps)
   {
      if (Mpi::Root()) { std::cout << "Time step " << timeStepper.StepIndex() << ", time = " << timeStepper.Time() << ", time elapsed = " << mfem::toc() << std::endl; }
      logger << "Time step " << timeStepper.StepIndex() << ", time = " << timeStepper.Time() << ", time elapsed = " << mfem::toc() << std::endl;

      const real_t residual = timeStepper.SolveTimeStep();

      const int n = timeStepper.StepIndex(); // StepIndex() is updated by timeStepper.

      timer.start("DataIO");
      if (n % vis_steps == 0 || n == num_steps)
      {
         pvdc.SetCycle(n);
         pvdc.SetTime(timeStepper.Time());
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