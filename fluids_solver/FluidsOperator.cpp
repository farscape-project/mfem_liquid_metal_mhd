#include "FluidsOperator.hpp"

FluidsOperator::FluidsOperator(Array<ParFiniteElementSpace *> &fes,
                            Array<Array<int> *> &ess_bdr,
                            Array<Array<int> *> &nat_bdr,
                            Array<int> &offsets,
                            int dim_,
                            double dt_)
   : TimeDependentOperator(fes[0]->GetTrueVSize() + fes[1]->GetTrueVSize()),
     block_trueOffsets(offsets),
     fluids_solver(),
     zero_vector(3),
     one_vector(3),
     one_vector_coef(one_vector),
     velocity_nbc_coeff(velocity_nbc),
     pressure_nbc_coeff(pressure_nbc),
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
   //nat_bdr.Copy(nat_bdr_marker);

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
   //ru->AddDomainIntegrator(new VectorDomainLFIntegrator(*velocity_DBC));
   ru->AddDomainIntegrator(new VectorDomainLFIntegrator(*zero_coeff));

   // Set up rhs for pressure solve.
   rp = new ParLinearForm(spaces[1]);
   //rp->AddDomainIntegrator(new DomainLFIntegrator(*pressure_DBC));
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

   int v_space_size = spaces[0]->GetTrueVSize();
   int p_space_size = spaces[1]->GetTrueVSize(); 


   // Project BCs onto grid functions.
   u_gf.ProjectBdrCoefficient(*velocity_DBC, *ess_bdr_marker[0]);
   p_gf.ProjectBdrCoefficient(*pressure_DBC, *ess_bdr_marker[1]);

   FkMat = new OperatorHandle();
   BMat = new OperatorHandle();
   BtMat = new OperatorHandle();

   Vector Ru, Rp, Xu_dummy, Ru_dummy;

   fk->FormLinearSystem(ess_tdof_u, u_gf, *ru, *FkMat, Xu_dummy, Ru);  

   b->FormRectangularLinearSystem(ess_tdof_u, ess_tdof_p,
                               u_gf, *rp, *BMat,
                               Xu_dummy, Rp);

   bT->FormRectangularLinearSystem(ess_tdof_p, ess_tdof_u,
                                p_gf, *ru, *BtMat,
                                Ru_dummy, Xu_dummy);


   A = new BlockOperator(block_trueOffsets);
   // Set F block for velocity.
   A->SetBlock(0,0, FkMat->Ptr()); 
   // Set coupling (B^T and B) blocks.
   A->SetBlock(0,1, BtMat->Ptr());
   A->SetBlock(1,0, BMat->Ptr());

   RHS = new BlockVector(block_trueOffsets);
   RHS->GetBlock(0) = Ru; 
   RHS->GetBlock(1) = Rp; 


   // IMPLEMENT PRECONDITIONER.
   /*P = new BlockDiagonalPreconditioner(block_trueOffsets);

   ParBilinearForm L(spaces[1]);
   L.AddDomainIntegrator(new DiffusionIntegrator());  
   L.Assemble();
   L.Finalize();
   //HypreParMatrix *LMat = L.ParallelAssemble();

   //SparseMatrix &FkMatPrec = fk->SpMat();
   //HypreParMatrix *FkMatPrec = fk->ParallelAssemble();

   LMat = new OperatorHandle();

   //invF = new DSmoother(FkMatPrec);
   invF = new HypreSmoother(*FkMat->As<HypreParMatrix>());
   //invF = new DSmoother(FkMat->Ptr());
   
   //invS = new DSmoother(LMat->Ptr());
   invS = new HypreSmoother(*LMat->As<HypreParMatrix>());

   P->SetDiagonalBlock(0, invF);
   P->SetDiagonalBlock(1, invS);*/


   fluids_solver.SetOperator(*A);
   //fluids_solver.SetPreconditioner(*P);
   
}



//void FluidsOperator::Mult(const Vector &X, Vector &dX_dt) const
void FluidsOperator::ImplicitSolve(const real_t dt,
                                const Vector &X, Vector &dX_dt)
{
   fluids_solver.Mult(*RHS, dX_dt); 

   // The solver actually calculates Xn+1, not dX/dt.  
   // The derivative is calculated below:
   dX_dt -= X;       // dX = Xn+1 - Xn
   dX_dt /= dt;      // dX/dt = (Xn+1 - Xn) / dt

}

FluidsOperator::~FluidsOperator()
{
   delete fk;
   delete b;
   delete ru;
   delete rp;
}


void FluidsOperator::EnforceDirichletBCs(BlockVector &X)
{
   int v_size = spaces[0]->GetTrueVSize();
   //int p_size = spaces[1]->GetTrueVSize();

   Vector u(X.GetData(), v_size);
   //Vector p(X.GetData() + v_size, p_size);

   ParGridFunction xu(spaces[0]);
   //ParGridFunction xp(spaces[1]);

   xu = 0.0;
   //xp = 0.0;

   xu.SetFromTrueDofs(u);
   //xp.SetFromTrueDofs(p);

   xu.ProjectBdrCoefficient(*velocity_DBC, *ess_bdr_marker[0]);
   //xp.ProjectBdrCoefficient(*pressure_DBC, *ess_bdr_marker[1]);

   xu.GetTrueDofs(u);
   //xp.GetTrueDofs(p);
}