#include "FluidsOperator.hpp"

FluidsOperator::FluidsOperator(Array<FiniteElementSpace *> &fes,
                            Array<Array<int> *> &ess_bdr,
                            Array<Array<int> *> &nat_bdr,
                            Array<int> &offsets,
                            int dim)
   : Operator(fes[0]->GetTrueVSize() + fes[1]->GetTrueVSize()),
     block_trueOffsets(offsets),
     fluids_solver(),
     zero_vector(3),
     one_vector(3),
     zero_vector_coef(zero_vector),
     one_vector_coef(one_vector),
     velocity_nbc_coeff(velocity_nbc),
     pressure_nbc_coeff(pressure_nbc)
{
   fes.Copy(spaces);

   ess_bdr.Copy(ess_bdr_marker);
   nat_bdr.Copy(nat_bdr_marker);

   zero_vector = 0.0;
   one_vector = 0.0;

   // Set up rhs for velocity solve.
   ru = new LinearForm(spaces[0]);
   ru->AddDomainIntegrator(new VectorDomainLFIntegrator(zero_vector_coef));
   // Natural boundary condition.
   ru->AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(velocity_nbc_coeff), *nat_bdr_marker[0]);

   // Set up rhs for pressure solve.
   rp = new LinearForm(spaces[1]);
   rp->AddDomainIntegrator(new DomainLFIntegrator(zero));
   // Natural boundary condition.
   rp->AddBoundaryIntegrator(new BoundaryLFIntegrator(pressure_nbc_coeff), *nat_bdr_marker[1]);

   FunctionCoefficient oneFunc([](const Vector &x) {
         return 1.0; 
   });

   // Simple forcing test.
   //rp->AddDomainIntegrator(new DomainLFIntegrator(oneFunc));

   // Set up mixed bilinear form for velocity and pressure coupling.
   b = new MixedBilinearForm(spaces[0],spaces[1]);
   b->AddDomainIntegrator(new VectorDivergenceIntegrator(neg_one));

   // Define ustar_n for convection integrators.
   ustar_n = new GridFunction(spaces[0]);
   ucoef = new VectorFunctionCoefficient(dim, u_exact);
   ustar_n->ProjectCoefficient(*ucoef);
   ustar_coef = new VectorGridFunctionCoefficient(ustar_n);

   // Bilinear form for the velocity solve.
   fk = new BilinearForm(spaces[0]);
   // Integrator for (v, v').
   fk->AddDomainIntegrator(new VectorMassIntegrator(vectorMassCoef));
   // Integrator for A_AL(v, v').
   fk->AddDomainIntegrator(new VectorDiffusionIntegrator(reciprocal_Re_coef));

   //VectorConvectionIntegratorCoefficient ConvIntCoeff(ustar_coef);
   //fk->AddDomainIntegrator(new VectorMassIntegrator(ConvIntCoeff));
    
   fk->AddDomainIntegrator(new VectorConvectionIntegrator(*ustar_coef,0.5));
   fk->AddDomainIntegrator(new ConservativeVectorConvectionIntegrator(*ustar_coef,0.5));

   // Outflow boundary term.
   FunctionCoefficient outflow_term(outflow_term_func);
   fk->AddBoundaryIntegrator(new VectorMassIntegrator(outflow_term));

   // Solve.
   //GMRESSolver gmres;
   //gmres.SetPreconditioner(*precond); 
   fluids_solver.SetRelTol(1e-8);
   fluids_solver.SetAbsTol(0.0);
   fluids_solver.SetMaxIter(500);
   fluids_solver.SetPrintLevel(1);
   //fluids_solver.SetPreconditioner(BlockDiagonalPreconditioner);
   fluids_solver.iterative_mode = false;  


   // Initialize the Jacobian preconditioner
   //PPreconditioner *jac_prec =
   //   new PPreconditioner(fes, *pressure_mass, block_trueOffsets);
   //j_prec = jac_prec;

}

// Solve the fluids GMRES problem.
void FluidsOperator::Solve(Vector &X)
{
   // Assume X is a BlockVector
   BlockVector &Xb = dynamic_cast<BlockVector&>(X);

   GridFunction xu_gf(spaces[0]), xp_gf(spaces[1]);
   xu_gf = 0.0; xp_gf = 0.0;
   //xu_gf.SetTrueVector();
   //xp_gf.SetTrueVector();

   xu_gf.SetFromTrueDofs(Xb.GetBlock(0));
   xp_gf.SetFromTrueDofs(Xb.GetBlock(1));

   // Copy true dofs into GridFunctions
   //xu_gf.MakeRef(spaces[0], Xb.GetBlock(0), 0);
   //xp_gf.MakeRef(spaces[1], Xb.GetBlock(1), 0);

   velocity_DBC = new VectorFunctionCoefficient(dim, velocity_dbc_vec_func);
   pressure_DBC = new FunctionCoefficient(pressure_dbc);

   // Project BCs onto grid functions.
   xu_gf.ProjectBdrCoefficient(*velocity_DBC, *ess_bdr_marker[0]);
   xp_gf.ProjectBdrCoefficient(*pressure_DBC, *ess_bdr_marker[1]);

   std::cout << "Xu_true:" << std::endl;
   Vector Xu_true(spaces[0]->GetTrueVSize());
   xu_gf.GetTrueDofs(Xu_true);
   Xu_true.Print();

   std::cout << "Xp_true:" << std::endl;
   Vector Xp_true(spaces[1]->GetTrueVSize());
   xp_gf.GetTrueDofs(Xp_true);
   Xp_true.Print();


   fk->Assemble(); fk->Finalize();
   b->Assemble(); b->Finalize();

   ru->Assemble();
   rp->Assemble();


   Array<int> ess_tdof_u, ess_tdof_p;
   spaces[0]->GetEssentialTrueDofs(*ess_bdr_marker[0], ess_tdof_u);
   spaces[1]->GetEssentialTrueDofs(*ess_bdr_marker[1], ess_tdof_p);

   SparseMatrix FkMat = fk->SpMat();
   SparseMatrix BMat = b->SpMat();


   // Eliminate rows for DBC and generate rhs subtraction for velocity DBCs.

   // Set up the true dofs for each grid function.
   Vector u_true(spaces[0]->GetTrueVSize());
   xu_gf.GetTrueDofs(u_true);

   u_true.Print();

   Vector p_true(spaces[1]->GetTrueVSize());
   xp_gf.GetTrueDofs(p_true);


   // Loop through essential bcs for velocity and eliminate rows in the Fk matrix and set 
   // diagonal entries to 1.0.
   Vector u_ess(u_true.Size());
   u_ess = 0.0;
   for (int i = 0; i < ess_tdof_u.Size(); i++)
   {
      int tdof_u = ess_tdof_u[i];
      // Set RHS to the Dirichlet value.
      u_ess(tdof_u) = u_true(tdof_u); 

      // Zero row
      FkMat.EliminateRow(tdof_u);
      // Set diagonal entry
      FkMat.Set(tdof_u, tdof_u, 1.0);
   }


   // Loop through essential bcs for velocity and eliminate columns in the b matrix.
   Vector b_ess_u(u_true.Size());
   for (int i = 0; i < ess_tdof_u.Size(); i++)
   {
      int tdof_u = ess_tdof_u[i];
      // Copy only constrained values.
      b_ess_u(tdof_u) = u_true(tdof_u);  
      
      // Zero row
      BMat.EliminateCol(tdof_u);
      // Set diagonal entry
      //BtMat->Set(tdof_u, tdof_u, 1.0);
   }

   // Loop through essential bcs for pressure and eliminate rows in the b matrix.
   Vector b_ess_p(p_true.Size());
   for (int i = 0; i < ess_tdof_p.Size(); i++)
   {
      int tdof_p = ess_tdof_p[i];
      // Copy only constrained values.
      b_ess_p(tdof_p) = p_true(tdof_p);  

      // Zero row
      BMat.EliminateRow(tdof_p);
      // Set diagonal entry
      //BtMat->Set(tdof_p, tdof_p, 1.0);
   }

   SparseMatrix *BtMat = Transpose(BMat);  // Bᵗ: pressure → velocity

   Vector BEss(p_true.Size());
   BMat.Mult(b_ess_u, BEss);

   Vector BtEss(u_true.Size());
   BtMat->Mult(b_ess_p, BtEss);

   Vector FkMatEss(u_true.Size());
   FkMat.Mult(u_ess, FkMatEss);

   Vector Ru(spaces[0]->GetTrueVSize()), Rp(spaces[1]->GetTrueVSize());
   Ru = ru->GetData();
   Rp = rp->GetData();

   // Subtract the essential boundary conditions from the rhs.
   // Ru = Ru - FkMat * u_ess - B^T * p_ess
   Ru -= FkMatEss;
   Ru -= BtEss;

   // Rp = Rp - B * u_ess
   Rp -= BEss;

   std::cout << "||Ru|| = " << Ru.Norml2() << std::endl;
   std::cout << "||Rp|| = " << Rp.Norml2() << std::endl;


   A = new BlockOperator(block_trueOffsets);
   // Set F block for velocity.
   A->SetBlock(0,0, &FkMat); 
   // Set coupling (B^T and B) blocks.
   A->SetBlock(0,1, BtMat);
   A->SetBlock(1,0, &BMat);

   RHS = new BlockVector(block_trueOffsets);
   RHS->GetBlock(0) = Ru; 
   RHS->GetBlock(1) = Rp; 

   BlockDiagonalPreconditioner M(block_trueOffsets);
   M.SetDiagonalBlock(0, new GSSmoother(FkMat)); // Velocity preconditioner

   fluids_solver.SetOperator(*A);
   fluids_solver.SetPreconditioner(M);
   fluids_solver.Mult(*RHS, X);

   
}

// Solve AX = RHS.
void FluidsOperator::Mult(const Vector &x, Vector &y) const
{
   fluids_solver.Mult(x, y);
}


FluidsOperator::~FluidsOperator()
{
   delete A;
   delete fk;
   delete b;
   delete ru;
   delete rp;
}
