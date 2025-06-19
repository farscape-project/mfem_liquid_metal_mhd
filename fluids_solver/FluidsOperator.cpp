#include "FluidsOperator.hpp"

FluidsOperator::FluidsOperator(Array<FiniteElementSpace *> &fes,
                            Array<Array<int> *> &ess_bdr,
                            Array<Array<int> *> &nat_bdr,
                            Array<int> &offsets,
                            int dim)
   : Operator(fes[0]->GetTrueVSize() + fes[1]->GetTrueVSize()),
     block_trueOffsets(offsets),
     xu_gf(fes[0]),
     xp_gf(fes[1]),
     fluids_solver(),
     ustar_n(fes[0]),
     ustar_coef(&ustar_n),
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

   // FIXME: xu_gf not getting to Solve properly???
   // ess_bdr not getting to Solve properly???

   // Define grid functions for velocity and pressure, respectively.
   //GridFunction xu_gf(spaces[0]), xp_gf(spaces[1]);
   //xu_gf.MakeRef(spaces[0], Vector(), 0);
   //xp_gf.MakeRef(spaces[1], Vector(), 0);

   // Initialise grid functions with zero, then with boundary conditions.
   xu_gf = 0.0; xp_gf = 0.0;
   //xu_gf.ProjectCoefficient(velocity_DBC);
   //xp_gf.ProjectCoefficient(pressure_DBC);

   // Project boundaries.  Choose between this and ProjectCoefficient above.
   //xu_gf.ProjectBdrCoefficient(velocity_DBC, *ess_bdr[0]);
   //xp_gf.ProjectBdrCoefficient(pressure_DBC, *ess_bdr[1]);

   xu_gf.SetTrueVector();
   xp_gf.SetTrueVector();

   // Set up rhs for velocity solve.
   ru = new LinearForm(spaces[0]);
   ru->AddDomainIntegrator(new VectorDomainLFIntegrator(zero_vector_coef));
   // Natural boundary condition.
   ru->AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(velocity_nbc_coeff), *nat_bdr_marker[0]);
   //ru.Assemble();

   // Set up rhs for pressure solve.
   rp = new LinearForm(spaces[1]);
   rp->AddDomainIntegrator(new DomainLFIntegrator(zero));
   // Natural boundary condition.
   rp->AddBoundaryIntegrator(new BoundaryLFIntegrator(pressure_nbc_coeff), *nat_bdr_marker[1]);

   FunctionCoefficient oneFunc([](const Vector &x) {
         return 1.0; // constant divergence source
   });

   // Simple forcing test.
   //rp->AddDomainIntegrator(new DomainLFIntegrator(oneFunc));
   //rp.Assemble();

   // Set up mixed bilinear form for velocity and pressure coupling.
   b = new MixedBilinearForm(spaces[0],spaces[1]);
   b->AddDomainIntegrator(new VectorDivergenceIntegrator(neg_one));
   //b.Assemble();
   //b.Finalize();

   // Define ustar_n for convection integrators.
   //GridFunction ustar_n(spaces[0]); // 0.5(3u_{n-1} - u_{n-2})
   VectorFunctionCoefficient ucoef(dim, u_exact);
   ustar_n.ProjectCoefficient(ucoef);
   //VectorGridFunctionCoefficient ustar_coef(&ustar_n);

   // Bilinear form for the velocity solve.
   //BilinearForm fk(spaces[0]);
   fk = new BilinearForm(spaces[0]);
   // Integrator for (v, v').
   fk->AddDomainIntegrator(new VectorMassIntegrator(vectorMassCoef));
   // Integrator for A_AL(v, v').
   fk->AddDomainIntegrator(new VectorDiffusionIntegrator(reciprocal_Re_coef));

   //VectorConvectionIntegratorCoefficient ConvIntCoeff(ustar_coef);
   //fk->AddDomainIntegrator(new VectorMassIntegrator(ConvIntCoeff));
    
   fk->AddDomainIntegrator(new VectorConvectionIntegrator(ustar_coef,0.5));
   fk->AddDomainIntegrator(new ConservativeVectorConvectionIntegrator(ustar_coef,0.5));

   // Outflow boundary term.
   FunctionCoefficient outflow_term(outflow_term_func);
   fk->AddBoundaryIntegrator(new VectorMassIntegrator(outflow_term));

   //fk->Assemble();
   //fk->Finalize();

   // Initialise solution and RHS vectors.
   //BlockVector X(block_trueOffsets), 
   //X = 0.0;
   

   // Solve.
   //GMRESSolver gmres;
   //gmres.SetPreconditioner(*precond); 
   fluids_solver.SetRelTol(1e-8);
   fluids_solver.SetAbsTol(0.0);
   fluids_solver.SetMaxIter(200);
   fluids_solver.SetPrintLevel(1);
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

   // Copy true dofs into GridFunctions
   xu_gf.MakeRef(spaces[0], Xb.GetBlock(0), 0);
   xp_gf.MakeRef(spaces[1], Xb.GetBlock(1), 0);

   velocity_DBC = new VectorFunctionCoefficient(dim, velocity_dbc_vec_func);
   pressure_DBC = new FunctionCoefficient(pressure_dbc);

   // Apply BCs.
   xu_gf.ProjectBdrCoefficient(*velocity_DBC, *ess_bdr_marker[0]);
   xp_gf.ProjectBdrCoefficient(*pressure_DBC, *ess_bdr_marker[1]);

   ru->Assemble();
   rp->Assemble();
   fk->Assemble();
   fk->Finalize();
   b->Assemble();
   b->Finalize();

   // TESTS
   std::cout << "||ru|| = " << ru->Norml2() << std::endl;
   std::cout << "||rp|| = " << rp->Norml2() << std::endl;

   SparseMatrix FkMat, P_dummy;
   Vector Xu, Ru, Xp, Rp;

   // TESTS
   std::cout << "ru.Size() = " << ru->Size() << std::endl;
   std::cout << "xu_gf.Size() = " << xu_gf.Size() << std::endl;
   std::cout << "ess_bdr_marker[0]->Size() = " << ess_bdr_marker[0]->Size() << std::endl;


   fk->FormLinearSystem(*ess_bdr_marker[0], xu_gf, *ru, FkMat, Xu, Ru);

   // Dummy bilinear form for applying pressure bcs.
   BilinearForm p_dummy(spaces[1]);
   p_dummy.Assemble();
   p_dummy.Finalize();
   p_dummy.FormLinearSystem(*ess_bdr_marker[1], xp_gf, *rp, P_dummy, Xp, Rp);

   // TESTS (begin)
   std::cout << "||xp_gf|| = " << xp_gf.Norml2() << std::endl;

   Vector test_vel(spaces[0]->GetTrueVSize()); test_vel.Randomize();
   Vector div_result(spaces[1]->GetTrueVSize());

   b->SpMat().Mult(test_vel, div_result);
   std::cout << "||B * random_u|| = " << div_result.Norml2() << std::endl;

   Vector test_pres(spaces[1]->GetTrueVSize());
   test_pres.Randomize();

   Vector grad_result(spaces[0]->GetTrueVSize());
   TransposeOperator Bt(b->SpMat());
   Bt.Mult(test_pres, grad_result);

   std::cout << "||Bᵗ * random_p|| = " << grad_result.Norml2() << std::endl;
   // TESTS (end)

   A = new BlockOperator(block_trueOffsets);
   // Set F block for velocity.
   A->SetBlock(0,0, &FkMat); 
   // Set coupling (B^T and B) blocks.
   Operator* BtOp = new TransposeOperator(b->SpMat());
   A->SetBlock(0,1, BtOp);
   A->SetBlock(1,0, &(b->SpMat()));

   RHS = new BlockVector(block_trueOffsets);
   RHS->GetBlock(0) = Ru; 
   RHS->GetBlock(1) = Rp; 

   // TESTS
   std::cout << "||Rp|| = " << RHS->GetBlock(1).Norml2() << std::endl;
   std::cout << "||Ru|| = " << RHS->GetBlock(0).Norml2() << std::endl;

   fluids_solver.SetOperator(*A);
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
