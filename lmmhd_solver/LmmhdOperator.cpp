#include "LmmhdOperator.hpp"

LmmhdOperator::LmmhdOperator(Array<ParFiniteElementSpace *> &fes,
                            Array<Array<int> *> &ess_bdr,
                            Array<int> &offsets,
                            int dim_,
                            double dt_)
   : TimeDependentOperator(fes[0]->GetTrueVSize() + fes[1]->GetTrueVSize() + fes[2]->GetTrueVSize() + fes[3]->GetTrueVSize()),
     block_trueOffsets(offsets),
     lmmhd_solver(),
     zero_vector(3),
     one_vector(3),
     one_vector_coef(one_vector),
     dim(dim_),
     dt(dt_)
{
   // Sets up the linear system for the coupled MHD solve:
   //
   //       [  Mj    G^T    K^T       0   ] [ xj   ]   [ rj   ]
   //       [  G     0      0         0   ] [ xphi ]   [ rphi ]
   //       [ -K     0      Fk       B^T  ] [ xu   ] = [ ru   ]
   //       [  0     0      B         0   ] [ xp   ]   [ rp   ]
   //
   // where:
   //   - Mj    : current-density bilinear form: (d,d')
   //   - G     : coupling between current and electric potential: -(div d, phi)
   //   - K     : coupling between current and velocity: (d, B x v')
   //   - Fk    : velocity bilinear form: 2/Tau (v, v') + O(u*_n; v, v') + A_AL(v, v')
   //   - B     : coupling between velocity and pressure: -(div v, q)

   fes.Copy(spaces);
   ess_bdr.Copy(ess_bdr_marker);

   zero_vector = 0.0;
   one_vector = 0.0;

   //massCoefValue = 1.0 / dt;
   massCoefValue = 1.0; // The mass coefficient is NOT divided by dt as 
   // this is taken into account in ImplicitSolve.
   massCoef = new ConstantCoefficient(massCoefValue);

   // Set Dirichlet boundary condition functions (not time-dependent).
   currentD_DBC = new VectorFunctionCoefficient(dim, currentD_dbc_vec_func);
   electPot_DBC = new FunctionCoefficient(electPot_dbc);
   pressure_DBC = new FunctionCoefficient(pressure_dbc);

   zero_coeff = new VectorFunctionCoefficient(dim, zero_func);
   zero_coef = new ConstantCoefficient(0.0);

   velocity_DBC = new VectorFunctionCoefficient(dim, velocity_dbc_vec_func);
   magnetic_field_coef = new VectorFunctionCoefficient(dim, magnetic_field_func);

   // Set up rhs for velocity solve.
   ru = new ParLinearForm(spaces[2]);
   ru->AddDomainIntegrator(new VectorDomainLFIntegrator(*zero_coeff));

   // Set up rhs for pressure solve.
   rp = new ParLinearForm(spaces[3]);
   rp->AddDomainIntegrator(new DomainLFIntegrator(*zero_coef));

   // Set up rhs for current density solve.
   rj = new ParLinearForm(spaces[0]);
   rj->AddDomainIntegrator(new VectorFEDomainLFIntegrator(*zero_coeff));

   // Set up rhs for electric potential solve.
   rphi = new ParLinearForm(spaces[1]);
   rphi->AddDomainIntegrator(new DomainLFIntegrator(*zero_coef));

   // Initisalise bilinear forms.
   fk = new ParBilinearForm(spaces[2]);
   b = new ParMixedBilinearForm(spaces[2],spaces[3]);
   bT = new ParMixedBilinearForm(spaces[3],spaces[2]);

   mj = new ParBilinearForm(spaces[0]);
   g = new ParMixedBilinearForm(spaces[0],spaces[1]);
   gT = new ParMixedBilinearForm(spaces[1],spaces[0]);
   k = new ParMixedBilinearForm(spaces[0],spaces[2]);
   //Ptmp = new ParBilinearForm(spaces[1]);

   B = new Vector(dim);
   *B = 0.0;
   (*B)(2) = 1.0;
   C = new CrossProductMatrixCoefficient(*B);

   // Set solver parameters.
   lmmhd_solver.SetRelTol(1e-6);
   lmmhd_solver.SetAbsTol(0.0);
   lmmhd_solver.SetMaxIter(1000);
   lmmhd_solver.SetPrintLevel(1);
   lmmhd_solver.iterative_mode = false;  

}

void LmmhdOperator::Update(const Vector &X)
{

   Array<int> ess_tdof_j, ess_tdof_phi, ess_tdof_u, ess_tdof_p;
   spaces[0]->GetEssentialTrueDofs(*ess_bdr_marker[0], ess_tdof_j);
   spaces[1]->GetEssentialTrueDofs(*ess_bdr_marker[1], ess_tdof_phi);
   spaces[2]->GetEssentialTrueDofs(*ess_bdr_marker[2], ess_tdof_u);
   spaces[3]->GetEssentialTrueDofs(*ess_bdr_marker[3], ess_tdof_p);


   ParGridFunction j_gf(spaces[0]), phi_gf(spaces[1]), u_gf(spaces[2]), p_gf(spaces[3]);
   j_gf = 0.0; phi_gf = 0.0; u_gf = 0.0; p_gf = 0.0;

   // Bilinear form for the velocity.
   delete fk;
   fk = new ParBilinearForm(spaces[2]);
   // Integrator for (v, v').
   fk->AddDomainIntegrator(new VectorMassIntegrator(*massCoef));
   // Integrator for A_AL(v, v').
   fk->AddDomainIntegrator(new VectorDiffusionIntegrator(reciprocal_Re_coef));
   // Integrator for O(u_n; v, v').
   fk->AddDomainIntegrator(new VectorConvectionIntegrator(*ustar_coef,0.5));
   fk->AddDomainIntegrator(new ConservativeVectorConvectionIntegrator(*ustar_coef,-0.5));

   // Mixed bilinear form for velocity and pressure coupling.
   delete b;
   b = new ParMixedBilinearForm(spaces[2],spaces[3]);
   b->AddDomainIntegrator(new VectorDivergenceIntegrator(neg_one));

   delete bT;
   bT = new ParMixedBilinearForm(spaces[3],spaces[2]);
   bT->AddDomainIntegrator(new GradientIntegrator(one));

   // Bilinear form for current density.
   delete mj;
   mj = new ParBilinearForm(spaces[0]);
   mj->AddDomainIntegrator(new VectorFEMassIntegrator(one));

   // Mixed bilinear form for current density and electric potential coupling.
   delete g;
   g = new ParMixedBilinearForm(spaces[0],spaces[1]);
   g->AddDomainIntegrator(new MixedScalarDivergenceIntegrator(neg_one));

   delete gT;
   gT = new ParMixedBilinearForm(spaces[1],spaces[0]);
   gT->AddDomainIntegrator(new MixedVectorGradientIntegrator(one));

   // Mixed bilinear form for current density and velocity coupling.
   delete k;
   k = new ParMixedBilinearForm(spaces[0],spaces[2]);
   k->AddDomainIntegrator(new VectorFEMassIntegrator(*C));

   fk->Assemble(); fk->Finalize();
   b->Assemble(); b->Finalize();
   bT->Assemble(); bT->Finalize();

   mj->Assemble(); mj->Finalize();
   g->Assemble(); g->Finalize();
   gT->Assemble(); gT->Finalize();
   k->Assemble(); k->Finalize();

   ru->Assemble();
   rp->Assemble();
   rj->Assemble();
   rphi->Assemble();

   // Project BCs onto grid functions and set up HypreParMatrices.
   j_gf.ProjectBdrCoefficient(*currentD_DBC, *ess_bdr_marker[0]);
   phi_gf.ProjectBdrCoefficient(*electPot_DBC, *ess_bdr_marker[1]);
   u_gf.ProjectBdrCoefficient(*velocity_DBC, *ess_bdr_marker[2]);
   p_gf.ProjectBdrCoefficient(*pressure_DBC, *ess_bdr_marker[3]);

   MjMat = new HypreParMatrix();
   FkMat = new HypreParMatrix();
   BMat = new HypreParMatrix();
   BtMat = new HypreParMatrix();
   GMat = new HypreParMatrix();
   GTMat = new HypreParMatrix();
   KMat = new HypreParMatrix();
   KtMat = new HypreParMatrix();

   Vector Rj, Rphi, Ru, Rp, Xu_dummy, Ru_dummy;

   fk->FormLinearSystem(ess_tdof_u, u_gf, *ru, *FkMat, Xu_dummy, Ru);  

   b->FormRectangularLinearSystem(ess_tdof_u, ess_tdof_p,
                               u_gf, *rp, *BMat,
                               Xu_dummy, Rp);

   bT->FormRectangularLinearSystem(ess_tdof_p, ess_tdof_u,
                                p_gf, *ru, *BtMat,
                                Ru_dummy, Xu_dummy);

   mj->FormLinearSystem(ess_tdof_j, j_gf, *rj, *MjMat, Xu_dummy, Rj);

   g->FormRectangularLinearSystem(ess_tdof_j, ess_tdof_phi,
                               j_gf, *rphi, *GMat,
                               Xu_dummy, Rphi);

   gT->FormRectangularLinearSystem(ess_tdof_phi, ess_tdof_j,
                                phi_gf, *rj, *GTMat,
                                Ru_dummy, Xu_dummy);

   k->FormRectangularLinearSystem(ess_tdof_j, ess_tdof_u,
                                j_gf, *ru, *KMat,
                                Ru_dummy, Xu_dummy);

   
   KtMat = KMat->Transpose();


   A = new BlockOperator(block_trueOffsets);
   A->SetBlock(0,0, MjMat);
   A->SetBlock(0,1, GTMat);
   A->SetBlock(1,0, GMat);
   //A->SetBlock(1,1, PtmpMat);
   // Set F block for velocity.
   A->SetBlock(2,2, FkMat); 
   // Set coupling (B^T and B) blocks.
   A->SetBlock(2,3, BtMat);
   A->SetBlock(3,2, BMat);
   // Set K blocks for coupling J and U.
   A->SetBlock(2,0, KMat);
   A->SetBlock(0,2, KtMat);


   RHS = new BlockVector(block_trueOffsets);
   RHS->GetBlock(0) = Rj;
   RHS->GetBlock(1) = Rphi;
   RHS->GetBlock(2) = Ru; 
   RHS->GetBlock(3) = Rp; 


   // Preconditioner.
   P = new BlockDiagonalPreconditioner(block_trueOffsets);

   // Current density preconditioner.
   dj = new ParBilinearForm(spaces[0]);
   dj->AddDomainIntegrator(new VectorFEMassIntegrator(one));
   dj->AddDomainIntegrator(new DivDivIntegrator(one));
   dj->Assemble(); dj->Finalize();

   DjMat = new HypreParMatrix();
   dj->FormLinearSystem(ess_tdof_j, j_gf, *rj, *DjMat, Xu_dummy, Rj);

   // Electric potential preconditioner.
   mphi = new ParBilinearForm(spaces[1]);
   mphi->AddDomainIntegrator(new MassIntegrator(neg_one));
   mphi->Assemble(); mphi->Finalize();

   MphiMat = new HypreParMatrix();
   mphi->FormLinearSystem(ess_tdof_p, p_gf, *rp, *MphiMat, Xu_dummy, Rp);

   // Pressure preconditioner.
   sp = new ParBilinearForm(spaces[3]);
   sp->AddDomainIntegrator(new DiffusionIntegrator(neg_one));  
   sp->Assemble(); sp->Finalize();

   SpMat = new HypreParMatrix();
   sp->FormLinearSystem(ess_tdof_p, p_gf, *rp, *SpMat, Xu_dummy, Rphi);

   invDj = new HypreSmoother(*DjMat);
   invMphi = new HypreSmoother(*MphiMat);
   invF = new HypreSmoother(*FkMat);
   invSp = new HypreSmoother(*SpMat);

   //P->SetDiagonalBlock(0, invDj);
   //P->SetDiagonalBlock(1, invMphi);
   P->SetDiagonalBlock(2, invF);
   P->SetDiagonalBlock(3, invSp);

   //P = new BlockTriangularPreconditioner(*FkMat, *BtMat, spaces, ess_tdof_p, v_space_size, p_space_size, dt);
   //P->SetOperator(*FkMat);

   lmmhd_solver.SetPreconditioner(*P);
   lmmhd_solver.SetOperator(*A);
   
}



void LmmhdOperator::ImplicitSolve(const real_t dt,
                                const Vector &X, Vector &dX_dt)
{
   lmmhd_solver.Mult(*RHS, dX_dt); 

   // The solver actually calculates Xn+1, not dX/dt.  
   // The derivative is calculated below:
   dX_dt -= X;       // dX = Xn+1 - Xn
   dX_dt /= dt;      // dX/dt = (Xn+1 - Xn) / dt

}

LmmhdOperator::~LmmhdOperator()
{
   delete fk;
   delete b;
   delete ru;
   delete rp;
}