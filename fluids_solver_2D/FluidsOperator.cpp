#include "FluidsOperator.hpp"

FluidsOperator::FluidsOperator(Array<ParFiniteElementSpace *> &fes,
                            Array<Array<int> *> &ess_bdr,
                            Array<int> &offsets,
                            int dim_,
                            double dt_)
   : TimeDependentOperator(fes[0]->GetTrueVSize() + fes[1]->GetTrueVSize()),
     block_trueOffsets(offsets),
     fluids_solver(),
     zero_vector(3),
     one_vector(3),
     one_vector_coef(one_vector),
     dim(dim_),
     dt(dt_)
{
   // Sets up solve for the following system:
   //    ( Fk  B^T ) ( xu ) = ( ru )
   //    ( B   0   ) ( xp )   ( rp )
   // where Fk is the velocity bilinear form, B is the mixed bilinear form, and
   // ru and rp are the right-hand sides for the velocity and pressure solves, where
   //     Fk <-> 2/Tau (v, v') + O(u_n; v, v') + A_AL(v, v')
   // and
   //     B <-> -div(v, q).

   fes.Copy(spaces);
   ess_bdr.Copy(ess_bdr_marker);

   zero_vector = 0.0;
   one_vector = 0.0;

   //massCoefValue = 1.0 / dt;
   massCoefValue = 1.0; // The mass coefficient is NOT divided by dt as 
   // this is taken into account in ImplicitSolve.
   massCoef = new ConstantCoefficient(massCoefValue);

   // Set Dirichlet boundary condition functions (not time-dependent).
   velocity_DBC = new VectorFunctionCoefficient(dim, velocity_dbc_vec_func);
   pressure_DBC = new FunctionCoefficient(pressure_dbc);

   zero_coeff = new VectorFunctionCoefficient(dim, zero_func);
   zero_coef = new ConstantCoefficient(0.0);

   // Set up rhs for velocity solve.
   ru = new ParLinearForm(spaces[0]);
   ru->AddDomainIntegrator(new VectorDomainLFIntegrator(*zero_coeff));

   // Set up rhs for pressure solve.
   rp = new ParLinearForm(spaces[1]);
   rp->AddDomainIntegrator(new DomainLFIntegrator(*zero_coef));

   // Initisalise bilinear forms.
   fk = new ParBilinearForm(spaces[0]);
   b = new ParMixedBilinearForm(spaces[0],spaces[1]);
   bT = new ParMixedBilinearForm(spaces[1],spaces[0]);

   // Set solver parameters.
   fluids_solver.SetRelTol(1e-6);
   fluids_solver.SetAbsTol(0.0);
   fluids_solver.SetMaxIter(20000);
   fluids_solver.SetPrintLevel(0);
   fluids_solver.iterative_mode = false;  

}

void FluidsOperator::Update(const Vector &X)
{

   Array<int> ess_tdof_u, ess_tdof_p;
   spaces[0]->GetEssentialTrueDofs(*ess_bdr_marker[0], ess_tdof_u);
   spaces[1]->GetEssentialTrueDofs(*ess_bdr_marker[1], ess_tdof_p);


   ParGridFunction u_gf(spaces[0]), p_gf(spaces[1]);
   u_gf = 0.0; p_gf = 0.0;

   // Set up the bilinear form for the velocity solve.
   delete fk;
   // Bilinear form for the velocity solve.
   fk = new ParBilinearForm(spaces[0]);
   // Integrator for (v, v').
   fk->AddDomainIntegrator(new VectorMassIntegrator(*massCoef));
   // Integrator for A_AL(v, v').
   fk->AddDomainIntegrator(new VectorDiffusionIntegrator(reciprocal_Re_coef));
   // Integrator for O(u_n; v, v').
   fk->AddDomainIntegrator(new VectorConvectionIntegrator(*ustar_coef,0.5));
   fk->AddDomainIntegrator(new ConservativeVectorConvectionIntegrator(*ustar_coef,-0.5));

   // Set up mixed bilinear form for velocity and pressure coupling.
   delete b;
   b = new ParMixedBilinearForm(spaces[0],spaces[1]);
   b->AddDomainIntegrator(new VectorDivergenceIntegrator(neg_one));

   delete bT;
   bT = new ParMixedBilinearForm(spaces[1],spaces[0]);
   bT->AddDomainIntegrator(new GradientIntegrator(one));

   fk->Assemble(); fk->Finalize();
   b->Assemble(); b->Finalize();
   bT->Assemble(); bT->Finalize();

   ru->Assemble();
   rp->Assemble();

   //int v_space_size = spaces[0]->GetTrueVSize();
   //int p_space_size = spaces[1]->GetTrueVSize(); 


   // Project BCs onto grid functions and set up HypreParMatrices.
   u_gf.ProjectBdrCoefficient(*velocity_DBC, *ess_bdr_marker[0]);
   //p_gf.ProjectBdrCoefficient(*pressure_DBC, *ess_bdr_marker[1]);

   /*std::cout << "Velocity gridFunction values:\n";
   u_gf.Print(std::cout); 

   std::cout << "Pressure gridFunction values:\n";
   p_gf.Print(std::cout);*/

   //std::cout << "pressure: ess_tdof_p size = " << ess_bdr_marker[1]->Size() << "\n";

   FkMat = new HypreParMatrix();
   BMat = new HypreParMatrix();
   BtMat = new HypreParMatrix();

   //Array<int> ess_tdof_p_single;
   //if (ess_tdof_p.Size() > 0) { ess_tdof_p_single.Append(ess_tdof_p[0]); }
   //else { std::cerr << "No pressure TDofs found to pin!\n"; }  

   Vector Ru, Rp;//, Xu_dummy, Ru_dummy;
   Xu_dummy = 0.0; Ru_dummy = 0.0;

   fk->FormLinearSystem(ess_tdof_u, u_gf, *ru, *FkMat, Xu_dummy, Ru);  

   b->FormRectangularLinearSystem(ess_tdof_u, ess_tdof_p,
                               u_gf, *rp, *BMat,
                               Xu_dummy, Rp);

   bT->FormRectangularLinearSystem(ess_tdof_p, ess_tdof_u,
                                p_gf, *ru, *BtMat,
                                Ru_dummy, Xu_dummy);


   A = new BlockOperator(block_trueOffsets);
   // Set F block for velocity.
   A->SetBlock(0,0, FkMat); 
   // Set coupling (B^T and B) blocks.
   A->SetBlock(0,1, BtMat);
   A->SetBlock(1,0, BMat);

   RHS = new BlockVector(block_trueOffsets);
   RHS->GetBlock(0) = Ru; 
   RHS->GetBlock(1) = Rp; 


   // Preconditioner.
   P = new BlockDiagonalPreconditioner(block_trueOffsets);

   s = new ParBilinearForm(spaces[1]);
   s->AddDomainIntegrator(new DiffusionIntegrator());  
   s->Assemble();
   s->Finalize();

   SMat = new HypreParMatrix();
   s->FormLinearSystem(ess_tdof_p, p_gf, *rp, *SMat, Xu_dummy, Rp);

   invF = new HypreSmoother(*FkMat);
   invS = new HypreSmoother(*SMat);

   P->SetDiagonalBlock(0, invF);
   P->SetDiagonalBlock(1, invS);

   //P = new BlockTriangularPreconditioner(*FkMat, *BtMat, spaces, ess_tdof_p, v_space_size, p_space_size, dt);
   //P->SetOperator(*FkMat);

   fluids_solver.SetPreconditioner(*P);
   fluids_solver.SetOperator(*A);
   
}



void FluidsOperator::ImplicitSolve(const real_t dt,
                                const Vector &X, Vector &dX_dt)
{
   fluids_solver.Mult(*RHS, dX_dt); 

   // The solver actually calculates Xn+1, not dX/dt.  
   // The derivative is calculated below:
   dX_dt -= X;       // dX = Xn+1 - Xn
   dX_dt /= dt;      // dX/dt = (Xn+1 - Xn) / dt

}

void FluidsOperator::UpdateXwithBCs(BlockVector &X)
{
   ParGridFunction p_gf(spaces[1]);

   bT->RecoverFEMSolution(Ru_dummy, Xu_dummy, p_gf);

   X.GetBlock(1) = p_gf;  
}

FluidsOperator::~FluidsOperator()
{
   delete fk;
   delete b;
   delete ru;
   delete rp;
}