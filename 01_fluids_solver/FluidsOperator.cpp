#include "FluidsOperator.hpp"

FluidsOperator::FluidsOperator(Array<FiniteElementSpace *> &fes,
                            Array<Array<int> *> &ess_bdr,
                            Array<Array<int> *> &nat_bdr,
                            Array<int> &offsets,
                            int dim)
   : Operator(fes[0]->GetTrueVSize() + fes[1]->GetTrueVSize()),
     fluids_solver(), block_trueOffsets(offsets)
{
    //Array<Vector *> RHS(2);
    //RHS = NULL; // Set all entries in the array

   fes.Copy(spaces);

   Vector zero_vector(3);
   Vector one_vector(3);
   zero_vector = 0.0;
   one_vector = 0.0;

   VectorConstantCoefficient zero_vector_coef(zero_vector);
   VectorConstantCoefficient one_vector_coef(one_vector);

   FunctionCoefficient velocity_nbc_coeff(velocity_nbc);
   FunctionCoefficient pressure_nbc_coeff(pressure_nbc);

    // Set up rhs for velocity solve.
    LinearForm ru(spaces[0]);
    ru.AddDomainIntegrator(new VectorDomainLFIntegrator(zero_vector_coef));
    // Natural boundary condition.
    ru.AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(velocity_nbc_coeff), *nat_bdr[0]);
    ru.Assemble();

    // Set up rhs for pressure solve.
    LinearForm rp(spaces[1]);
    rp.AddDomainIntegrator(new DomainLFIntegrator(zero));
    // Natural boundary condition.
    rp.AddBoundaryIntegrator(new BoundaryLFIntegrator(pressure_nbc_coeff), *nat_bdr[1]);
    rp.Assemble();

    // Set up mixed bilinear form for velocity and pressure coupling.
    MixedBilinearForm b(spaces[0],spaces[1]);
    b.AddDomainIntegrator(new VectorDivergenceIntegrator(neg_one));
    b.Assemble();
    b.Finalize();

    // Define ustar_n for convection integrators.
    GridFunction ustar_n(spaces[0]); // 0.5(3u_{n-1} - u_{n-2})
    VectorFunctionCoefficient ucoef(dim, u_exact);
    ustar_n.ProjectCoefficient(ucoef);
    VectorGridFunctionCoefficient ustar_coef(&ustar_n);

    // Bilinear form for the velocity solve.
    BilinearForm fk(spaces[0]);
    // Integrator for (v, v').
    fk.AddDomainIntegrator(new VectorMassIntegrator(vectorMassCoef));
    // Integrator for A_AL(v, v').
    fk.AddDomainIntegrator(new VectorDiffusionIntegrator(reciprocal_Re_coef));
    
    fk.AddDomainIntegrator(new VectorConvectionIntegrator(ustar_coef,0.5));
    fk.AddDomainIntegrator(new ConservativeVectorConvectionIntegrator(ustar_coef,0.5));

    // Outflow boundary term.
    FunctionCoefficient outflow_term(outflow_term_func);
    fk.AddBoundaryIntegrator(new VectorMassIntegrator(outflow_term));

    fk.Assemble();
    fk.Finalize();

    // Needs grid function.  And what do to about X, at the moment it comes from Solve?
    SparseMatrix Fk;
    fk.FormLinearSystem(*ess_bdr[0], yu, ru, Fk, X[0], RHS[0]);

    // Needs a dummy solver.
    //sp.FormLinearSystem(&ess_bdr[1], yp, rp, Sp, Xp, Rp);

    // Initialise solution and RHS vectors.
    //BlockVector X(block_trueOffsets), 
    //X = 0.0;
    RHS = new BlockVector(block_trueOffsets);
    RHS->GetBlock(0) = ru; 
    RHS->GetBlock(1) = rp; 


    BlockOperator A(block_trueOffsets);

   // Set F block for velocity.
    A.SetBlock(0,0, &fk); 

    // Set coupling (B^T and B) blocks.
    Operator* bt = new TransposeOperator(b);
    A.SetBlock(0,1, bt);
    A.SetBlock(1,0, &b);

    // Solve.
    //GMRESSolver gmres;
    fluids_solver.SetOperator(A);
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

// Solve the Newton system
void FluidsOperator::Solve(Vector &X) const
{
    //Vector rhs;
    fluids_solver.Mult(*RHS, X);
}

// compute: y = H(x,p)
void FluidsOperator::Mult(const Vector &x, Vector &y) const
{
   //Hform->Mult(k, y);
   fluids_solver.Mult(x, y);
}

// Compute the Jacobian from the nonlinear form
//Operator &FluidsOperator::GetGradient(const Vector &xp) const
//{
//   return Hform->GetGradient(xp);
//}

FluidsOperator::~FluidsOperator()
{
   //delete Hform;
   //delete pressure_mass;
   //delete j_solver;
   //delete j_prec;
}
