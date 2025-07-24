#include "FluidsOperator.hpp"

FluidsOperator::FluidsOperator(Array<ParFiniteElementSpace *> &fes,
                            Array<Array<int> *> &ess_bdr,
                            Array<Array<int> *> &nat_bdr,
                            Array<int> &offsets,
                            int dim,
                            double dt_)
   : TimeDependentOperator(fes[0]->GetTrueVSize() + fes[1]->GetTrueVSize()),
     block_trueOffsets(offsets),
     fluids_solver(),
     zero_vector(3),
     one_vector(3),
     zero_vector_coef(zero_vector),
     one_vector_coef(one_vector),
     velocity_nbc_coeff(velocity_nbc),
     pressure_nbc_coeff(pressure_nbc),
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
   nat_bdr.Copy(nat_bdr_marker);

   zero_vector = 0.0;
   one_vector = 0.0;

   // Set up rhs for velocity solve.
   ru = new ParLinearForm(spaces[0]);
   ru->AddDomainIntegrator(new VectorDomainLFIntegrator(zero_vector_coef));
   // Natural boundary condition.
   ru->AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(velocity_nbc_coeff), *nat_bdr_marker[0]);

   // Set up rhs for pressure solve.
   rp = new ParLinearForm(spaces[1]);
   rp->AddDomainIntegrator(new DomainLFIntegrator(zero));
   // Natural boundary condition.
   rp->AddBoundaryIntegrator(new BoundaryLFIntegrator(pressure_nbc_coeff), *nat_bdr_marker[1]);

   FunctionCoefficient oneFunc([](const Vector &x) {
         return 1.0; 
   });

   // Initisalise bilinear forms.
   fk = new ParBilinearForm(spaces[0]);
   b = new ParMixedBilinearForm(spaces[0],spaces[1]);

   // Define ustar_n for convection integrators.
   //ustar_n = new GridFunction(spaces[0]);
   //ucoef = new VectorFunctionCoefficient(dim, u_exact);
   //ustar_n->ProjectCoefficient(*ucoef);
   //VectorGridFunctionCoefficient *ustar_coef = nullptr;

   // Coefficient for mass matrix defined as 2/Tau where Tau is the time step.
   ConstantCoefficient vectorMassCoef(2.0 / dt);

   // Solve.
   fluids_solver.SetRelTol(1e-6);
   fluids_solver.SetAbsTol(0.0);
   fluids_solver.SetMaxIter(1000);
   fluids_solver.SetPrintLevel(0);
   fluids_solver.iterative_mode = false;  

   //BlockDiagonalPreconditioner P(block_trueOffsets);


   // Initialize the Jacobian preconditioner
   //PPreconditioner *jac_prec =
   //   new PPreconditioner(fes, *pressure_mass, block_trueOffsets);
   //j_prec = jac_prec;

}

void FluidsOperator::Update(const Vector &X)
{
   // Set up the bilinear form for the velocity solve.
   delete fk;
   // Bilinear form for the velocity solve.
   fk = new ParBilinearForm(spaces[0]);
   // Integrator for (v, v').
   fk->AddDomainIntegrator(new VectorMassIntegrator(vectorMassCoef));
   // Integrator for A_AL(v, v').
   fk->AddDomainIntegrator(new VectorDiffusionIntegrator(reciprocal_Re_coef));

   //VectorConvectionIntegratorCoefficient ConvIntCoeff(ustar_coef);
   //fk->AddDomainIntegrator(new VectorMassIntegrator(ConvIntCoeff));
    
   //ustar_coef = new VectorGridFunctionCoefficient(&u_star);

   fk->AddDomainIntegrator(new VectorConvectionIntegrator(*ustar_coef,0.5));
   fk->AddDomainIntegrator(new ConservativeVectorConvectionIntegrator(*ustar_coef,0.5));

   // Outflow boundary term.
   FunctionCoefficient outflow_term(outflow_term_func);
   fk->AddBoundaryIntegrator(new VectorMassIntegrator(outflow_term));

   
   // Set up mixed bilinear form for velocity and pressure coupling.
   delete b;
   b = new ParMixedBilinearForm(spaces[0],spaces[1]);
   b->AddDomainIntegrator(new VectorDivergenceIntegrator(neg_one));

   int v_space_size = spaces[0]->GetTrueVSize();
   int p_space_size = spaces[1]->GetTrueVSize(); 

   // Extract velocity and pressure parts from the input vector X.
   Vector u_part(X.GetData(), v_space_size);            
   Vector p_part(X.GetData() + v_space_size, p_space_size);

   ParGridFunction xu_gf(spaces[0]), xp_gf(spaces[1]);
   xu_gf = 0.0; xp_gf = 0.0;
   //xu_gf.SetTrueVector();
   //xp_gf.SetTrueVector();

   //xu_gf.SetFromTrueDofs(Xb->GetBlock(0));
   //xp_gf.SetFromTrueDofs(Xb->GetBlock(1));

   xu_gf.SetFromTrueDofs(u_part);
   xp_gf.SetFromTrueDofs(p_part);

   // Copy true dofs into GridFunctions
   //xu_gf.MakeRef(spaces[0], Xb.GetBlock(0), 0);
   //xp_gf.MakeRef(spaces[1], Xb.GetBlock(1), 0);

   velocity_DBC = new VectorFunctionCoefficient(dim, velocity_dbc_vec_func);
   pressure_DBC = new FunctionCoefficient(pressure_dbc);

   // Project BCs onto grid functions.
   xu_gf.ProjectBdrCoefficient(*velocity_DBC, *ess_bdr_marker[0]);
   xp_gf.ProjectBdrCoefficient(*pressure_DBC, *ess_bdr_marker[1]);


   //std::cout << "Xu_true:" << std::endl;
   //Vector Xu_true(spaces[0]->GetTrueVSize());
   //xu_gf.GetTrueDofs(Xu_true);
   //Xu_true.Print();

   //std::cout << "Xp_true:" << std::endl;
   //Vector Xp_true(spaces[1]->GetTrueVSize());
   //xp_gf.GetTrueDofs(Xp_true);
   //Xp_true.Print();

   fk->Assemble(); fk->Finalize();
   b->Assemble(); b->Finalize();

   ru->Assemble();
   rp->Assemble();

   Array<int> ess_tdof_u, ess_tdof_p;
   spaces[0]->GetEssentialTrueDofs(*ess_bdr_marker[0], ess_tdof_u);
   spaces[1]->GetEssentialTrueDofs(*ess_bdr_marker[1], ess_tdof_p);

   serialFkMat = &(fk->SpMat());
   BMat = &(b->SpMat());

   // Manually handle the essential boundary conditions (DBC) for velocity and pressure.
   // Eliminate rows for DBC and generate rhs subtraction for velocity DBCs.

   // Set up the true dofs for each grid function.
   Vector u_true(spaces[0]->GetTrueVSize());
   xu_gf.GetTrueDofs(u_true);

   Vector p_true(spaces[1]->GetTrueVSize());
   xp_gf.GetTrueDofs(p_true);

   //std::cout << "u_true:" << std::endl;
   //u_true.Print();  // Look for values near 1.0 at inlet DOFs


   // Loop through essential bcs for velocity and eliminate rows in the Fk matrix and set 
   // diagonal entries to 1.0.
   Vector u_ess(u_true.Size());
   u_ess = 0.0;
   for (int i = 0; i < ess_tdof_u.Size(); i++)
   {
      int tdof_u = ess_tdof_u[i];
      // Set RHS to the Dirichlet value.
      u_ess(tdof_u) = u_true(tdof_u); 

      // Zero row.  TODO: Double check if this should be EliminateRow or EliminateRowCol.
      serialFkMat->EliminateRowCol(tdof_u);
      // Set diagonal entry
      serialFkMat->Set(tdof_u, tdof_u, 1.0);
   }

   /*std::cout << "Essential velocity BCs applied (tdof : value):" << std::endl;
   for (int i = 0; i < ess_tdof_u.Size(); i++)
   {
      int tdof_u = ess_tdof_u[i];
      std::cout << "  DOF " << tdof_u << " : " << u_true(tdof_u) << std::endl;
   }*/
      

   // Loop through essential bcs for velocity and eliminate columns in the b matrix.
   Vector b_ess_u(u_true.Size());
   for (int i = 0; i < ess_tdof_u.Size(); i++)
   {
      int tdof_u = ess_tdof_u[i];
      // Copy only constrained values.
      b_ess_u(tdof_u) = u_true(tdof_u);  
      
      // Zero row
      BMat->EliminateCol(tdof_u);
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
      BMat->EliminateRow(tdof_p);
      // Set diagonal entry
      //BtMat->Set(tdof_p, tdof_p, 1.0);
   }


   //SparseMatrix *BtMat = Transpose(*BMat);  // B^T: pressure → velocity
   if (BtMat) { delete BtMat; } 
      BtMat = Transpose(*BMat);

   Vector BEss(p_true.Size());
   BMat->Mult(b_ess_u, BEss);

   Vector BtEss(u_true.Size());
   BtMat->Mult(b_ess_p, BtEss);

   Vector FkMatEss(u_true.Size());
   serialFkMat->Mult(u_ess, FkMatEss);

   //Vector Ru(spaces[0]->GetTrueVSize()), Rp(spaces[1]->GetTrueVSize());
   //Ru = ru->GetData();
   //Rp = rp->GetData();
   Vector Ru(*ru), Rp(*rp);

   // Scaling RHS by tau and adding mass matrix contribution.
   //Ru *= tau;
   //Ru += Mu_old;

   // Subtract the essential boundary conditions from the rhs.
   // Ru = Ru - FkMat * u_ess - B^T * p_ess
   Ru -= FkMatEss;
   Ru -= BtEss;

   // Rp = Rp - B * u_ess
   Rp -= BEss;

   std::cout << "||Ru|| = " << Ru.Norml2() << std::endl;
   std::cout << "||Rp|| = " << Rp.Norml2() << std::endl;

   HypreParMatrix *FkMat = fk->ParallelAssemble();

   A = new BlockOperator(block_trueOffsets);
   // Set F block for velocity.
   A->SetBlock(0,0, FkMat); 
   // Set coupling (B^T and B) blocks.
   A->SetBlock(0,1, BtMat);
   A->SetBlock(1,0, BMat);

   RHS = new BlockVector(block_trueOffsets);
   RHS->GetBlock(0) = Ru; 
   RHS->GetBlock(1) = Rp; 

   // Create preconditioner if it doesn't exist.  Update value of Fk (dependent on last value of velocity) 
   // if it already exists.
   /*if (!P) 
   {
      P = new BlockTriangularPreconditioner(*FkMat, *BtMat, spaces[1], v_space_size, p_space_size, vectorMassCoef, dt);
      P->Update(*FkMat);
      fluids_solver.SetPreconditioner(*P);
   }
   else
   {
      P->Update(*FkMat);
   }*/

   /*P = new BlockDiagonalPreconditioner(block_trueOffsets);
   P->SetDiagonalBlock(0, new GSSmoother(*FkMat)); // Velocity preconditioner

   m_p = new BilinearForm(spaces[1]);
   m_p->AddDomainIntegrator(new MassIntegrator(alpha1_coeff));
   m_p->Assemble(); m_p->Finalize();
   MpMat = &(m_p->SpMat());

   s_p = new BilinearForm(spaces[1]);
   s_p->AddDomainIntegrator(new DiffusionIntegrator(vectorMassCoef));
   s_p->Assemble(); s_p->Finalize();
   SpMat = &(s_p->SpMat());

   LMat = Add(*MpMat, *SpMat);  

   P->SetDiagonalBlock(1, new GSSmoother(*LMat));*/

   fluids_solver.SetOperator(*A);
   

}



void FluidsOperator::Mult(const Vector &X, Vector &dX_dt) const
//void FluidsOperator::ImplicitSolve(const real_t dt,
//                                const Vector &X, Vector &dX_dt)
// Solve the fluids GMRES problem.
//void FluidsOperator::ImplicitSolve(Vector &X)
{
   fluids_solver.Mult(*RHS, dX_dt);  
}

// Solve AX = RHS.
//void FluidsOperator::Mult(const Vector &x, Vector &y) const
//{
//   fluids_solver.Mult(x, y);
//}


FluidsOperator::~FluidsOperator()
{
   delete fk;
   delete b;
   delete ru;
   delete rp;
}
