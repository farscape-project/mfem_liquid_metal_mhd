#include "LmmhdOperator.hpp"

LmmhdOperator::LmmhdOperator(Array<ParFiniteElementSpace *> &fes,
                            Array<Array<int> *> &ess_bdr,
                            Array<int> &offsets,
                            int dim_,
                            real_t dt_)
   : TimeDependentOperator(fes[0]->GetTrueVSize() + fes[1]->GetTrueVSize() + fes[2]->GetTrueVSize() + fes[3]->GetTrueVSize()),
     block_trueOffsets(offsets),
     lmmhd_solver(),
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

   lmmhd_solver = new GMRESSolver(MPI_COMM_WORLD);
   fes.Copy(spaces);
   ess_bdr.Copy(ess_bdr_marker);

   // Set Dirichlet boundary condition functions (not time-dependent).
   currentD_DBC = new VectorFunctionCoefficient(dim, currentD_dbc_vec_func);
   electPot_DBC = new FunctionCoefficient(electPot_dbc);
   pressure_DBC = new FunctionCoefficient(pressure_dbc);

   zeroCoeff = new ConstantCoefficient(0.0);
   vectorZeroCoeff = new VectorFunctionCoefficient(dim, zero_func);

   velocity_DBC = new VectorFunctionCoefficient(dim, velocity_dbc_vec_func);

   // Set up magnetic field vector B.
   B = new Vector(dim);
   *B = 0.0;
   (*B)(0) = Bx;
   (*B)(1) = By;
   (*B)(2) = Bz;

   /// Set up coefficients for f bilinear forms in Update. 
   // Set up Matrix coefficient for the integrator (Kv, Kv') where 
   // Kv = B x v for fk in preconditioner.
   DenseMatrix BxVBxV(dim);
   real_t Bnorm2 = (*B) * (*B);
   // Build M = |B|^2 I - B B^T (equivalent to K where K(v) = (B x v, B x v') ).
   for (int i = 0; i < dim; i++)
   {
      for (int j = 0; j < dim; j++)
         BxVBxV(i,j) = (i == j ? Bnorm2 : 0.0) - (*B)(i) * (*B)(j);
   }
   DenseMatrix kappaBxVBxV(dim);
   kappaBxVBxV = BxVBxV;
   kappaBxVBxV *= kappa_val;
   fkBxVBxVcoeff = new MatrixConstantCoefficient(kappaBxVBxV);

   //massCoefValue = 1.0 / dt;
   // The mass coefficient is NOT divided by dt as 
   // this is taken into account in ImplicitSolve.
   massCoeffValue = 1.0; 
   fMassCoeff = new ConstantCoefficient(massCoeffValue);
   fReciprocalReCoeff = new ConstantCoefficient(reciprocal_Re);


   /// Set up RHS linear forms.
   // Set up rhs for velocity solve.
   ru = new ParLinearForm(spaces[2]);
   ru->AddDomainIntegrator(new VectorDomainLFIntegrator(*vectorZeroCoeff));
   ru->Assemble();

   // Set up rhs for pressure solve.
   rp = new ParLinearForm(spaces[3]);
   rp->AddDomainIntegrator(new DomainLFIntegrator(*zeroCoeff));
   rp->Assemble();

   // Set up rhs for current density solve.
   rj = new ParLinearForm(spaces[0]);
   rj->AddDomainIntegrator(new VectorFEDomainLFIntegrator(*vectorZeroCoeff));
   rj->Assemble();

   // Set up rhs for electric potential solve.
   rphi = new ParLinearForm(spaces[1]);
   rphi->AddDomainIntegrator(new DomainLFIntegrator(*zeroCoeff));
   rphi->Assemble();


   /// Set up bilinear forms.
   // Initialise bilinear forms.
   fu = new ParBilinearForm(spaces[2]);
   fk = new ParBilinearForm(spaces[2]);

   b = new ParMixedBilinearForm(spaces[2],spaces[3]);
   bT = new ParMixedBilinearForm(spaces[3],spaces[2]);
   mj = new ParBilinearForm(spaces[0]);
   g = new ParMixedBilinearForm(spaces[0],spaces[1]);
   gT = new ParMixedBilinearForm(spaces[1],spaces[0]);
   k = new ParMixedBilinearForm(spaces[0],spaces[2]);

   
   // Dirichlet boundary conditions.
   Array<int> ess_tdof_j, ess_tdof_phi, ess_tdof_u, ess_tdof_p;
   spaces[0]->GetEssentialTrueDofs(*ess_bdr_marker[0], ess_tdof_j);
   spaces[1]->GetEssentialTrueDofs(*ess_bdr_marker[1], ess_tdof_phi);
   spaces[3]->GetEssentialTrueDofs(*ess_bdr_marker[3], ess_tdof_p);

   ParGridFunction j_gf(spaces[0]), phi_gf(spaces[1]), u_gf(spaces[2]), p_gf(spaces[3]);
   j_gf = 0.0; phi_gf = 0.0; p_gf = 0.0;

   // Project BCs onto grid functions.
   j_gf.ProjectBdrCoefficient(*currentD_DBC, *ess_bdr_marker[0]);
   phi_gf.ProjectBdrCoefficient(*electPot_DBC, *ess_bdr_marker[1]);
   p_gf.ProjectBdrCoefficient(*pressure_DBC, *ess_bdr_marker[3]);


   // Mixed bilinear form for velocity and pressure coupling.
   bCoeff = new ConstantCoefficient(-1.0);
   //cout << "bCoeff value = " << bCoeff->constant << endl;
   b->AddDomainIntegrator(new VectorDivergenceIntegrator(*bCoeff));
   b->Assemble(); b->Finalize();

   //bTCoeff = new ConstantCoefficient(1.0);
   bTCoeff = new ConstantCoefficient(-1.0);
   //cout << "bTCoeff value = " << bTCoeff->constant << endl;
   bT->AddDomainIntegrator(new GradientIntegrator(*bTCoeff));
   bT->Assemble(); bT->Finalize();

   // Bilinear form for current density.
   mjCoeff = new ConstantCoefficient(kappa_val);
   //cout << "mjCoeff value = " << mjCoeff->constant << endl;
   mj->AddDomainIntegrator(new VectorFEMassIntegrator(*mjCoeff));
   mj->Assemble(); mj->Finalize();

   // Mixed bilinear form for current density and electric potential coupling.
   gCoeff = new ConstantCoefficient(-kappa_val);
   //cout << "gCoeff value = " << gCoeff->constant << endl;
   g->AddDomainIntegrator(new MixedScalarDivergenceIntegrator(*gCoeff));
   g->Assemble(); g->Finalize();

   //gTCoeff = new ConstantCoefficient(kappa_val);
   gTCoeff = new ConstantCoefficient(-kappa_val);
   //cout << "gTCoeff value = " << gTCoeff->constant << endl;
   gT->AddDomainIntegrator(new MixedVectorGradientIntegrator(*gTCoeff));
   gT->Assemble(); gT->Finalize();

   // Mixed bilinear form for current density and velocity coupling.
   // VectorFEMassIntegrator(lambda) applies (lambda d, v').  In order to 
   // apply (d, B x v') we rearrange the identity to (-B x d, v') using the 
   // identities a . (b x c) = c . (a x b) and (a x b) = - (b x a).  We also 
   // multiply by kappa to include Hartmann number scaling.
   kCoeffVec = new Vector(dim);
   *kCoeffVec = 0.0;
   *kCoeffVec -= *B;
   *kCoeffVec *= kappa_val;
   *kCoeffVec *= -1.0;
   //std::cout << "kCoeffVec: " << (*kCoeffVec)(0) << ", " << (*kCoeffVec)(1) << ", " << (*kCoeffVec)(2) << std::endl;
   kCoeff = new CrossProductMatrixCoefficient(*kCoeffVec);
   k->AddDomainIntegrator(new VectorFEMassIntegrator(*kCoeff));
   k->Assemble(); k->Finalize();

   MjMat = new HypreParMatrix();
   BMat = new HypreParMatrix();
   BtMat = new HypreParMatrix();
   GMat = new HypreParMatrix();
   GTMat = new HypreParMatrix();
   KMat = new HypreParMatrix();
   KtMat = new HypreParMatrix();

   Vector Rj, Rphi, Ru, Rp, Xu_dummy, Ru_dummy;

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

   // Transpose of KMat with negative sign as KMat is negative and KtMat positive.
   KtMat = KMat->Transpose();
   (*KtMat) *= -1.0;

   // Set up operator matrix.
   A = new BlockOperator(block_trueOffsets);
   A->SetBlock(0,0, MjMat);
   A->SetBlock(0,1, GTMat);
   A->SetBlock(1,0, GMat);
   // Set coupling (B^T and B) blocks.
   A->SetBlock(2,3, BtMat);
   A->SetBlock(3,2, BMat);
   // Set K blocks for coupling J and U.
   A->SetBlock(2,0, KMat);
   A->SetBlock(0,2, KtMat);

   // Set RHS.
   RHS = new BlockVector(block_trueOffsets);
   RHS->GetBlock(0) = Rj;
   RHS->GetBlock(1) = Rphi;
   RHS->GetBlock(3) = Rp; 


   /// Define integrators for preconditioner.
   // Current density preconditioner.
   djCoeff = new ConstantCoefficient(kappa_val); // Including kappa here although not present in algorithm 4.1.
   //cout << "djCoeff value = " << djCoeff->constant << endl;
   dj = new ParBilinearForm(spaces[0]);
   dj->AddDomainIntegrator(new VectorFEMassIntegrator(*djCoeff));  
   dj->AddDomainIntegrator(new DivDivIntegrator(*djCoeff));  
   dj->Assemble(); dj->Finalize();

   // Electric potential preconditioner.
   mphiCoeff = new ConstantCoefficient(kappa_val); // Including kappa here although not present in algorithm 4.1.
   //mphiCoeff = new ConstantCoefficient(-kappa_val); // Including kappa here although not present in algorithm 4.1.
   //cout << "mphiCoeff value = " << mphiCoeff->constant << endl;
   mphi = new ParBilinearForm(spaces[1]);
   mphi->AddDomainIntegrator(new MassIntegrator(*mphiCoeff));
   mphi->Assemble(); mphi->Finalize();

   // Pressure preconditioner (part 1).
   mpCoeff = new ConstantCoefficient(1.0);
   //cout << "mpCoeff value = " << mpCoeff->constant << endl;
   mp = new ParBilinearForm(spaces[3]);
   mp->AddDomainIntegrator(new MassIntegrator(*mpCoeff));
   mp->Assemble(); mp->Finalize();

   // Pressure preconditioner (part 2).
   spCoeff = new ConstantCoefficient(1.0);
   //cout << "spCoeff value = " << spCoeff->constant << endl;
   sp = new ParBilinearForm(spaces[3]);
   sp->AddDomainIntegrator(new DiffusionIntegrator(*spCoeff));
   sp->Assemble(); sp->Finalize();


   DjMat = new HypreParMatrix();
   MphiMat = new HypreParMatrix();
   MpMat = new HypreParMatrix();
   SpMat = new HypreParMatrix();
   
   dj->FormLinearSystem(ess_tdof_j, j_gf, *rj, *DjMat, Xu_dummy, Rj);
   mphi->FormLinearSystem(ess_tdof_p, p_gf, *rp, *MphiMat, Xu_dummy, Rp);
   mp->FormLinearSystem(ess_tdof_p, p_gf, *rp, *MpMat, Xu_dummy, Rphi);
   sp->FormLinearSystem(ess_tdof_p, p_gf, *rp, *SpMat, Xu_dummy, Rphi);


   // Preconditioner.
   P = new LiPreconditioner(spaces, block_trueOffsets);

   P->SetPressurePreconditioner(MpMat, SpMat);
   P->SetElectricPotentialPreconditioner(MphiMat);
   P->SetCurrentDensityPreconditioner(DjMat, GTMat, KtMat);

   // Set solver parameters.
   lmmhd_solver->SetRelTol(1e-4);
   lmmhd_solver->SetAbsTol(0.0);
   lmmhd_solver->SetMaxIter(500);
   lmmhd_solver->SetPrintLevel(1);
   lmmhd_solver->iterative_mode = false;  

}

void LmmhdOperator::Update(const Vector &X)
{
   Array<int> ess_tdof_u;
   spaces[2]->GetEssentialTrueDofs(*ess_bdr_marker[2], ess_tdof_u);

   ParGridFunction u_gf(spaces[2]);
   u_gf = 0.0;

   // Project BCs onto grid functions and set up HypreParMatrices.
   u_gf.ProjectBdrCoefficient(*velocity_DBC, *ess_bdr_marker[2]);

   // Bilinear form for the velocity.
   delete fu;
   fu = new ParBilinearForm(spaces[2]);
   // Integrator for (v, v').
   fu->AddDomainIntegrator(new VectorMassIntegrator(*fMassCoeff));
   // Integrator for A_AL(v, v').
   fu->AddDomainIntegrator(new VectorDiffusionIntegrator(*fReciprocalReCoeff));
   // Integrator for O(u_n; v, v').  ADD BOUNDARY TERM HERE.
   fu->AddDomainIntegrator(new VectorConvectionIntegrator(*ustar_coef,0.5));
   fu->AddDomainIntegrator(new ConservativeVectorConvectionIntegrator(*ustar_coef,-0.5));
   fu->Assemble(); fu->Finalize();

   FuMat = new HypreParMatrix();
   
   Vector Ru, Xu_dummy, Ru_dummy;
   fu->FormLinearSystem(ess_tdof_u, u_gf, *ru, *FuMat, Xu_dummy, Ru);  

   // Set F block and RHS for velocity.
   A->SetBlock(2,2, FuMat); 
   RHS->GetBlock(2) = Ru; 


   /// Preconditioner component.
   // Bilinear form for velocity preconditioner.
   delete fk;
   fk = new ParBilinearForm(spaces[2]);
   // Integrator for (v, v').
   fk->AddDomainIntegrator(new VectorMassIntegrator(*fMassCoeff));
   // Integrator for A_AL(v, v').
   fk->AddDomainIntegrator(new VectorDiffusionIntegrator(*fReciprocalReCoeff));
   // Integrator for O(u_n; v, v').
   fk->AddDomainIntegrator(new VectorConvectionIntegrator(*ustar_coef,0.5));
   fk->AddDomainIntegrator(new ConservativeVectorConvectionIntegrator(*ustar_coef,-0.5));
   fk->AddDomainIntegrator(new VectorMassIntegrator(*fkBxVBxVcoeff));
   fk->Assemble(); fk->Finalize();

   FkMat = new HypreParMatrix();

   fk->FormLinearSystem(ess_tdof_u, u_gf, *ru, *FkMat, Xu_dummy, Ru_dummy);  

   P->SetVelocityPreconditioner(FkMat, BtMat);

   lmmhd_solver->SetPreconditioner(*P);
   lmmhd_solver->SetOperator(*A);
   
}



void LmmhdOperator::ImplicitSolve(const real_t dt,
                                const Vector &X, Vector &dX_dt)
{
   lmmhd_solver->Mult(*RHS, dX_dt); 

   // The solver actually calculates Xn+1, not dX/dt.  
   // The derivative is calculated below:
   dX_dt -= X;       // dX = Xn+1 - Xn
   dX_dt /= dt;      // dX/dt = (Xn+1 - Xn) / dt

}

LmmhdOperator::~LmmhdOperator()
{
   delete fu;
   delete b;
   delete ru;
   delete rp;
}