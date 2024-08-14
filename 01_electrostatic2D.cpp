//
// Compile with: make 01_electrostatic2D
//
// Sample run:   01_electrostatic2D
//               01_electrostatic2D -m square.msh
//
// Description:  This example code solves a simple 2D/3D system of equations
//               for current density J and electric scalar potential phi
//
//                                 k*J + grad phi = f,
//                                 - div J        = g,
//
//               for k = 1, f = (0, -1), and g = 0 with insulating boundary 
//               conditions on left and right walls and conductiong boundary
//               conditions on top and bottom.
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

// Define the forcing terms / boundary conditions.
void fFun(const Vector & x, Vector & f);
real_t gFun(const Vector & x);
void current_dirichlet_bc(const Vector & x, Vector & u);
real_t voltage_bc(const Vector & x);

int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Parse command-line options.
   //const char *mesh_file = "./mesh/square_-1_to_1.msh";
   const char *mesh_file = "./mesh/square_0_to_1.msh";
   int order = 1;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();
   if (dim < 2 || dim > 3)
   {
      mfem::mfem_error("Mesh dimension must be 2 or 3.");
   }

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 10,000
   //    elements.
   {
      int ref_levels = (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
      ref_levels = 2;

      std::cout << "ref_levels = " << ref_levels << std::endl;
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 5. Define a finite element space on the mesh. Here we use the
   //    Raviart-Thomas finite elements of the specified order.
   FiniteElementCollection *hdiv_coll(new RT_FECollection(order, dim));
   FiniteElementCollection *l2_coll(new L2_FECollection(order, dim));

   FiniteElementSpace *R_space = new FiniteElementSpace(mesh, hdiv_coll);
   FiniteElementSpace *W_space = new FiniteElementSpace(mesh, l2_coll);

   // 6. Define the BlockStructure of the problem, i.e. define the array of
   //    offsets for each variable. The last component of the Array is the sum
   //    of the dimensions of each block.
   Array<int> block_offsets(3); // number of variables + 1
   block_offsets[0] = 0;
   block_offsets[1] = R_space->GetVSize();
   block_offsets[2] = W_space->GetVSize();
   block_offsets.PartialSum();

   std::cout << "***********************************************************\n";
   std::cout << "dim(R) = " << block_offsets[1] - block_offsets[0] << "\n";
   std::cout << "dim(W) = " << block_offsets[2] - block_offsets[1] << "\n";
   std::cout << "dim(R+W) = " << block_offsets.Last() << "\n";
   std::cout << "***********************************************************\n";

   // 7. Define the coefficients and rhs of the PDE.
   ConstantCoefficient one(1.0);
   ConstantCoefficient neg_one(-1.0);
   VectorFunctionCoefficient fcoeff(dim, fFun);
   FunctionCoefficient gcoeff(gFun);

   // 8. Allocate memory (x, rhs) for the solution and the right hand
   //    side.  Define the GridFunction J,phi for the finite element solution and
   //    linear forms fform and gform for the right hand side.  The data
   //    allocated by x and rhs are passed as a reference to the grid functions
   //    (J,phi) and the linear forms (fform, gform).
   MemoryType mt = device.GetMemoryType();
   BlockVector x(block_offsets, mt), rhs(block_offsets, mt);
   x = 0.0;

   LinearForm *fform(new LinearForm);
   fform->Update(R_space, rhs.GetBlock(0), 0);
   fform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff));
   //fform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(currentNeumannBC));
   fform->Assemble();
   fform->SyncAliasMemory(rhs);

   LinearForm *gform(new LinearForm);
   gform->Update(W_space, rhs.GetBlock(1), 0);
   gform->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
   gform->Assemble();
   gform->SyncAliasMemory(rhs);

   // 8a. Create non-zero Dirichlet boundary condition.

   // Choose tagged boundaries in cuboid.msh to apply condition to.
   Array<int> ess_bdr_lr(mesh->bdr_attributes.Max());
   ess_bdr_lr = 0;
   ess_bdr_lr[0] = 0;   // Top boundary
   ess_bdr_lr[1] = 1;   // Right boundary
   ess_bdr_lr[2] = 0;   // Bottom boundary
   ess_bdr_lr[3] = 1;   // Left boundary

   Array<int> ess_bdr_tb(mesh->bdr_attributes.Max());
   ess_bdr_tb = 0;
   ess_bdr_tb[0] = 1;   // Top boundary
   ess_bdr_tb[1] = 0;   // Right boundary
   ess_bdr_tb[2] = 1;   // Bottom boundary
   ess_bdr_tb[3] = 0;   // Left boundary


   // Project current boundary conditions defined in current_bc to grid function.
   GridFunction J_boundary;
   J_boundary.MakeRef(R_space, x.GetBlock(0), 0);
   VectorFunctionCoefficient J_coeff(dim, current_dirichlet_bc);
   J_boundary.ProjectCoefficient(J_coeff);
  

   GridFunction phi_boundary;
   phi_boundary.MakeRef(W_space, x.GetBlock(1), 0);
   FunctionCoefficient phi_coeff(voltage_bc);
   phi_boundary.ProjectCoefficient(phi_coeff);
   

   // 9. Assemble the finite element matrices for the Darcy operator
   //
   //                            D = [ a  b^T ]
   //                                [ b   0  ]
   //     where:
   //
   //     a(\vec{x},\vec{y}) = \int_\Omega \vec{x} \cdot \vec{y} d\Omega,
   //     b(x,\vec{y})   = -\int_\Omega \div y x d\Omega.


   BilinearForm *aBilForm(new BilinearForm(R_space));
   aBilForm->AddDomainIntegrator(new VectorFEMassIntegrator(one));
   aBilForm->Assemble();
   // Dirichlet BC.s
   aBilForm->EliminateEssentialBC(ess_bdr_lr, J_boundary, rhs.GetBlock(0));
   aBilForm->EliminateEssentialBC(ess_bdr_tb, phi_boundary, rhs.GetBlock(0));
   aBilForm->Finalize();


   MixedBilinearForm *bBilForm(new MixedBilinearForm(R_space, W_space));
   bBilForm->AddDomainIntegrator(new VectorFEDivergenceIntegrator(neg_one));
   bBilForm->Assemble();
   // Dirichlet BC.
   bBilForm->EliminateTrialDofs(ess_bdr_lr, J_boundary, rhs.GetBlock(1));
   bBilForm->EliminateTrialDofs(ess_bdr_tb, phi_boundary, rhs.GetBlock(1));
   bBilForm->Finalize();

   BlockOperator darcyOp(block_offsets);

   TransposeOperator *Bt = NULL;

   SparseMatrix &M(aBilForm->SpMat());
   SparseMatrix &B(bBilForm->SpMat());
   //B *= -1.;
   Bt = new TransposeOperator(&B);

   darcyOp.SetBlock(0,0, &M);
   darcyOp.SetBlock(0,1, Bt);
   darcyOp.SetBlock(1,0, &B);


   // 10. Construct the operators for preconditioner
   //
   //                 P = [ diag(M)         0         ]
   //                     [  0       B diag(M)^-1 B^T ]
   //
   //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
   //     pressure Schur Complement
   SparseMatrix *MinvBt = NULL;
   Vector Md(aBilForm->Height());

   BlockDiagonalPreconditioner darcyPrec(block_offsets);
   Solver *invM, *invS;
   SparseMatrix *S = NULL;

   M.GetDiag(Md);
   Md.HostReadWrite();
      
   MinvBt = Transpose(B);

   for (int i = 0; i < Md.Size(); i++)
   {
      MinvBt->ScaleRow(i, 1./Md(i));
   }

   S = Mult(B, *MinvBt);

   invM = new DSmoother(M);

#ifndef MFEM_USE_SUITESPARSE
      invS = new GSSmoother(*S);
#else
      invS = new UMFPackSolver(*S);
#endif

   invM->iterative_mode = false;
   invS->iterative_mode = false;

   darcyPrec.SetDiagonalBlock(0, invM);
   darcyPrec.SetDiagonalBlock(1, invS);

   // 11. Solve the linear system with MINRES.
   //     Check the norm of the unpreconditioned residual.
   int maxIter(10000);
   real_t rtol(1.e-8);
   real_t atol(1.e-10);

   chrono.Clear();
   chrono.Start();
   MINRESSolver solver;
   solver.SetAbsTol(atol);
   solver.SetRelTol(rtol);
   solver.SetMaxIter(maxIter);
   solver.SetOperator(darcyOp);
   solver.SetPreconditioner(darcyPrec);
   solver.SetPrintLevel(1);
   
   solver.Mult(rhs, x);
   if (device.IsEnabled()) { x.HostRead(); }
   chrono.Stop();

   if (solver.GetConverged())
   {
      std::cout << "MINRES converged in " << solver.GetNumIterations()
                << " iterations with a residual norm of "
                << solver.GetFinalNorm() << ".\n";
   }
   else
   {
      std::cout << "MINRES did not converge in " << solver.GetNumIterations()
                << " iterations. Residual norm is " << solver.GetFinalNorm()
                << ".\n";
   }
   std::cout << "MINRES solver took " << chrono.RealTime() << "s.\n";

   // 12. Create the grid functions J and phi and assign relevant blocks
   //     from solution x.
   GridFunction J, phi;
   J.MakeRef(R_space, x.GetBlock(0), 0);
   phi.MakeRef(W_space, x.GetBlock(1), 0);

   
   // 13. Save data in the ParaView format
   ParaViewDataCollection paraview_dc("current_solve", mesh);
   paraview_dc.SetPrefixPath("data");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetCycle(0);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetTime(0.0); // set the time
   paraview_dc.RegisterField("current density",&J);
   paraview_dc.RegisterField("electric scalar potential",&phi);
   paraview_dc.Save();

   // 13. Save the mesh and the solution. This output can be viewed later using
   //     GLVis: "glvis -m square.mesh -g sol_u.gf" or "glvis -m ex5.mesh -g
   //     sol_p.gf".
   {
      ofstream mesh_ofs("data/square.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);

      ofstream J_ofs("data/sol_J.gf");
      J_ofs.precision(8);
      J.Save(J_ofs);

      ofstream phi_ofs("data/sol_phi.gf");
      phi_ofs.precision(8);
      phi.Save(phi_ofs);
   }

   // 14. Free the used memory.
   delete fform;
   delete gform;
   delete invM;
   delete invS;
   delete S;
   delete Bt;
   delete MinvBt;
   delete aBilForm;
   delete bBilForm;
   delete W_space;
   delete R_space;
   delete l2_coll;
   delete hdiv_coll;
   delete mesh;

   return 0;
}


// Define functions for r.h.s. and boundary conditions.

void fFun(const Vector & x, Vector & f)
{
   
   f(0) = 0.0;
   f(1) = -1.0;

   if (x.Size() == 3)
   {
      f(2) = 0.0;
   }
}

real_t gFun(const Vector & x)
{
   return 0;
}

void current_dirichlet_bc(const Vector & x, Vector & u)
{
   real_t xi(x(0));
   real_t yi(x(1));

   u(0) = 0.0;
   u(1) = 0.0;
 
}

real_t voltage_bc(const Vector & x)
{
   return 0.0;
}
