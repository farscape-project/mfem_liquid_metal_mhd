#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;

void visualize(ParaViewDataCollection &paraview_dc, int order, GridFunction *field, 
   const char *field_name = NULL);

void checkpoint(int num);
real_t pressure_dbc(const Vector & x);
void velocity_dbc_vec_func(const Vector & x, Vector & f);

// Define constants.
real_t Re(1.0);
real_t reciprocal_Re(1 / Re);
real_t alpha(1.0); // alpha = 1 (default).
real_t alpha1(alpha + reciprocal_Re);
real_t neg_alpha1(-alpha1);
real_t tau(0.1);

// Define ConstantCoefficients.
ConstantCoefficient zero(0.0);
ConstantCoefficient one(1.0);
ConstantCoefficient neg_one(-1.0);
ConstantCoefficient reciprocal_Re_coef(reciprocal_Re);
ConstantCoefficient vectorMassCoef(2.0 / tau);

int main(int argc, char *argv[])
{
   // Define VectorConstantCoefficients.
   Vector zero_vector(3), one_vector(3);
   zero_vector = 0.0;
   VectorConstantCoefficient zero_vector_coef(zero_vector);
   one_vector = 1.0;
   VectorConstantCoefficient one_vector_coef(one_vector);
   
   Vector forcing_vector(3);
   forcing_vector = 0.0;
   forcing_vector[0] = 1.0;
   VectorConstantCoefficient forcing_vector_coef(forcing_vector);

   // Set fe_space orders.
   int order_pressure = 1;
   int order_velocity;
   order_velocity = order_pressure + 1;

   // Generate mesh.
   Mesh mesh = Mesh::MakeCartesian2D(20, 8, mfem::Element::Type::QUADRILATERAL, true, 5.0, 1.0);
   int dim = mesh.Dimension();   

   // ----------------------------------------------------------------------------
   // Finite Element Spaces.
   // ----------------------------------------------------------------------------
   // H1 continuous Lagrange finite elements of given order for pressure.
   H1_FECollection pressure_fec(order_pressure, dim);
   FiniteElementSpace pressure_fespace(&mesh, &pressure_fec);

   // H1 continuous Lagrange finite elements of given order (order_pressure + 1)
   // for velocity.
   H1_FECollection velocity_fec(order_velocity, dim);
   FiniteElementSpace velocity_fespace(&mesh, &velocity_fec, dim);

   Array<FiniteElementSpace *> spaces(2);
   spaces[0] = &velocity_fespace;
   spaces[1] = &pressure_fespace;

   int v_space_size = velocity_fespace.GetTrueVSize();
   int p_space_size = pressure_fespace.GetTrueVSize();


   // ----------------------------------------------------------------------------
   // Define locations of Dirichlet boundaries.
   // ----------------------------------------------------------------------------

   VectorFunctionCoefficient velocity_DBC(dim, velocity_dbc_vec_func);
   FunctionCoefficient pressure_DBC(pressure_dbc);

   Array<int> ess_boundary_marker_pressure, ess_boundary_marker_velocity; 

   // Essential (Dirichlet) boundary conditions for velocity.
   ess_boundary_marker_velocity.SetSize(velocity_fespace.GetMesh()->bdr_attributes.Max());
   ess_boundary_marker_velocity = 0;
   // Dirichlet boundary conditions for velocity.
   ess_boundary_marker_velocity[0] = 1; // Top
   ess_boundary_marker_velocity[1] = 0; // Outlet
   ess_boundary_marker_velocity[2] = 1; // Bottom
   ess_boundary_marker_velocity[3] = 1; // Inlet

   // Essential (Dirichlet) boundary conditions for pressure.
   ess_boundary_marker_pressure.SetSize(pressure_fespace.GetMesh()->bdr_attributes.Max());
   ess_boundary_marker_pressure = 0; 
   // Dirichlet boundary condition for pressure.
   ess_boundary_marker_pressure[0] = 0; // Top
   ess_boundary_marker_pressure[1] = 1; // Outlet
   ess_boundary_marker_pressure[2] = 0; // Bottom
   ess_boundary_marker_pressure[3] = 0; // Inlet
   

   Array<Array<int> *> ess_bdr(2);
   ess_bdr[0] = &ess_boundary_marker_velocity;
   ess_bdr[1] = &ess_boundary_marker_pressure;

   Array<int> ess_tdof_u, ess_tdof_p;
   pressure_fespace.GetEssentialTrueDofs(ess_boundary_marker_pressure, ess_tdof_p);
   velocity_fespace.GetEssentialTrueDofs(ess_boundary_marker_velocity, ess_tdof_u);


   // Define block structure of the solution vector.
   Array<int> block_trueOffsets(3);
   block_trueOffsets[0] = 0;
   block_trueOffsets[1] = velocity_fespace.GetTrueVSize();
   block_trueOffsets[2] = pressure_fespace.GetTrueVSize();
   block_trueOffsets.PartialSum();


   // ----------------------------------------------------------------------------
   // Initialise solution vector and GridFunctions.
   // ----------------------------------------------------------------------------
   BlockVector X(block_trueOffsets);
   X = 0;
   
   GridFunction u_gf(&velocity_fespace);
   GridFunction p_gf(&pressure_fespace);
   u_gf = 0.0;
   p_gf = 0.0;



   // ----------------------------------------------------------------------------
   // Set up (bi)linear forms.
   // ----------------------------------------------------------------------------

   // Set up rhs (incl. Dirichlet BC) for velocity solve.
   VectorFunctionCoefficient fcoeff(dim, velocity_dbc_vec_func);
   LinearForm ru(&velocity_fespace);
   //ru.AddDomainIntegrator(new VectorDomainLFIntegrator(fcoeff));
   ru.AddDomainIntegrator(new VectorDomainLFIntegrator(zero_vector_coef));
   ru.Assemble();

   // Set up rhs (incl. Dirichlet BC) for pressure solve.
   FunctionCoefficient gcoeff(pressure_dbc);
   LinearForm rp(&pressure_fespace);
   //rp.AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
   rp.AddDomainIntegrator(new DomainLFIntegrator(zero));
   rp.Assemble();

   MixedBilinearForm b(&velocity_fespace,&pressure_fespace);
   b.AddDomainIntegrator(new VectorDivergenceIntegrator(neg_one));
   b.Assemble();
   b.Finalize();

   MixedBilinearForm bT(&pressure_fespace,&velocity_fespace);
   bT.AddDomainIntegrator(new GradientIntegrator(one));
   bT.Assemble();
   bT.Finalize();

   // Bilinear form for the velocity solve.
   BilinearForm fk(&velocity_fespace);
   fk.AddDomainIntegrator(new VectorDiffusionIntegrator(reciprocal_Re_coef));
   fk.Assemble(); 
   fk.Finalize();
   

   // ----------------------------------------------------------------------------
   // Apply Dirichlet boundary conditions and set up linear system.
   // ----------------------------------------------------------------------------
   u_gf.ProjectBdrCoefficient(velocity_DBC, ess_boundary_marker_velocity);
   p_gf.ProjectBdrCoefficient(pressure_DBC, ess_boundary_marker_pressure);

   OperatorHandle FkMat, BMat, BtMat;
   Vector Ru, Rp, Xu_dummy, Ru_dummy;

   fk.FormLinearSystem(ess_tdof_u, u_gf, ru, FkMat, Xu_dummy, Ru);  

   b.FormRectangularLinearSystem(ess_tdof_u, ess_tdof_p,
                               u_gf, rp, BMat,
                               Xu_dummy, Rp);

   bT.FormRectangularLinearSystem(ess_tdof_p, ess_tdof_u,
                                p_gf, ru, BtMat,
                                Ru_dummy, Xu_dummy);


   // ----------------------------------------------------------------------------
   // Set up Block Operator (matrices) and right hand side vector.
   // ----------------------------------------------------------------------------

   BlockOperator A(block_trueOffsets);
   // Set F block for velocity.
   A.SetBlock(0,0, FkMat.Ptr()); 
   // Set coupling (B^T and B) blocks.
   A.SetBlock(0,1, BtMat.Ptr());
   A.SetBlock(1,0, BMat.Ptr());
 

   BlockVector RHS(block_trueOffsets);
   RHS.GetBlock(0) = Ru; 
   RHS.GetBlock(1) = Rp;


   // ----------------------------------------------------------------------------
   // Define preconditioner.
   // ----------------------------------------------------------------------------
   BlockDiagonalPreconditioner blockPrec(block_trueOffsets);
   Solver *invF, *invS;

   BilinearForm L(&pressure_fespace);
   L.AddDomainIntegrator(new DiffusionIntegrator());  
   L.Assemble();
   L.Finalize();
   SparseMatrix &LMat = L.SpMat();

   SparseMatrix &FkMatPrec = fk.SpMat();

   invF = new DSmoother(FkMatPrec);
   invS = new DSmoother(LMat);

   blockPrec.SetDiagonalBlock(0, invF);
   blockPrec.SetDiagonalBlock(1, invS);


   // ----------------------------------------------------------------------------
   // Solver.
   // ----------------------------------------------------------------------------
   GMRESSolver fluids_solver;
   fluids_solver.SetRelTol(1e-6);
   fluids_solver.SetAbsTol(0.0);
   fluids_solver.SetMaxIter(20000);
   fluids_solver.SetPrintLevel(1);
   fluids_solver.iterative_mode = true;
   fluids_solver.SetOperator(A);
   fluids_solver.SetPreconditioner(blockPrec);

   fluids_solver.Mult(RHS, X);  

   
   u_gf.SetFromTrueDofs(X.GetBlock(0));
   p_gf.SetFromTrueDofs(X.GetBlock(1));

   //fk.RecoverFEMSolution(X.GetBlock(0), RHS.GetBlock(0), u_gf);
   //b.RecoverFEMSolution(X.GetBlock(1), RHS.GetBlock(1), p_gf);

   u_gf.ProjectBdrCoefficient(velocity_DBC, ess_boundary_marker_velocity);
   p_gf.ProjectBdrCoefficient(pressure_DBC, ess_boundary_marker_pressure);


   // Set up visualisation in Paraview.
   ParaViewDataCollection paraview_dc("navier_stokes", &mesh);
   paraview_dc.SetPrefixPath("data");
   visualize(paraview_dc, order_velocity, &u_gf, "velocity");
   visualize(paraview_dc, order_pressure, &p_gf, "pressure");


   return 0;
}




real_t pressure_dbc(const Vector & x)
{
   return 0.0;
}


void velocity_dbc_vec_func(const Vector & x, Vector & f)
{
   f = 0.0;

   real_t r_mid = 0.5;
   real_t u_max = 1.0;
   
   if (x(0) < 1e-6)
   { // Parabolic condition in x-direction at inlet.
      f(0) = u_max * (1. - ((x(1) - r_mid)*(x(1) - r_mid)) / (r_mid * r_mid));
      f(1) = 0.0;
   }
   else
   { // Zero on top and bottom boundaries.
      f(0) = 0.0;
      f(1) = 0.0;
   }    

   if (x.Size() == 3)
   {
      f(2) = 0.0;
   }
}

void checkpoint(int num)
{
   cout << "**********************************************" << endl;
   cout << "**************** CHECKPOINT " << num << " ****************" << endl;
   cout << "**********************************************" << endl;
   cout << endl;
}

// Inline visualization
void visualize(ParaViewDataCollection &paraview_dc, int order, GridFunction *field, const char *field_name)
{

   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetCycle(0);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetTime(0.0); // set the time

   // Export field data.
   paraview_dc.RegisterField(field_name,field);

   paraview_dc.Save();
}
