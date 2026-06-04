#include "LmmhdOperator.hpp"

LmmhdOperator::LmmhdOperator(Array<ParFiniteElementSpace *> &fes,
                            Array<Array<int> *> &ess_bdr,
                            Array<int> &offsets,
                            int dim_,
                            real_t dt_,
                            int debug_,
                            int DIRECTSOLVE_)
   : TimeDependentOperator(fes[0]->GetTrueVSize() + fes[1]->GetTrueVSize() + fes[2]->GetTrueVSize() + fes[3]->GetTrueVSize()),
     block_trueOffsets(offsets),
     magnetics(fes[0],fes[1]),
     fluids(fes[2],fes[3]),
     coupling(fes[0],fes[2]),
     liprec(fes[0],fes[1],fes[2],fes[3]),
     ru(fes[2]),
     u_gf(fes[2]),
     lmmhd_solver(),
     dim(dim_),
     dt(dt_),
     debug(debug_),
     DIRECTSOLVE(DIRECTSOLVE_)
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
   
   if (DIRECTSOLVE == 0)
   {
      lmmhd_solver = std::make_unique<FGMRESSolver>(MPI_COMM_WORLD);
      dynamic_cast<FGMRESSolver*>(lmmhd_solver.get())->SetRelTol(1e-4);
      dynamic_cast<FGMRESSolver*>(lmmhd_solver.get())->SetAbsTol(1e-8);;
      dynamic_cast<FGMRESSolver*>(lmmhd_solver.get())->SetMaxIter(500);
      dynamic_cast<FGMRESSolver*>(lmmhd_solver.get())->SetPrintLevel(1);
      dynamic_cast<FGMRESSolver*>(lmmhd_solver.get())->SetKDim(500);
      dynamic_cast<FGMRESSolver*>(lmmhd_solver.get())->iterative_mode = true;
   }
   else
   {
      lmmhd_solver = std::make_unique<MUMPSSolver>(MPI_COMM_WORLD);
      dynamic_cast<MUMPSSolver*>(lmmhd_solver.get())->SetPrintLevel(2);
   }

   prec_ortho_solver = new OrthoSolver(MPI_COMM_WORLD);
   
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

   massCoeffValue = 2.0;
   // The mass coefficient is NOT divided by dt as 
   // this is taken into account in ImplicitSolve.
   //massCoeffValue = 2.0; 
   fMassCoeff = new ConstantCoefficient(massCoeffValue);
   fReciprocalReCoeff = new ConstantCoefficient(reciprocal_Re);


   /// Set up RHS linear forms.
   // Set up rhs for velocity solve.
   ru.AddDomainIntegrator(new VectorDomainLFIntegrator(*vectorZeroCoeff));
   ru.Assemble();

   // Set up rhs for pressure solve.
   ParLinearForm rp(spaces[3]);
   rp.AddDomainIntegrator(new DomainLFIntegrator(*zeroCoeff));
   rp.Assemble();

   // Set up rhs for current density solve.
   ParLinearForm rj(spaces[0]);
   rj.AddDomainIntegrator(new VectorFEDomainLFIntegrator(*vectorZeroCoeff));
   rj.Assemble();

   // Set up rhs for electric potential solve.
   ParLinearForm rphi(spaces[1]);
   rphi.AddDomainIntegrator(new DomainLFIntegrator(*zeroCoeff));
   rphi.Assemble();

   
   // Dirichlet boundary conditions.
   Array<int> ess_tdof_j, ess_tdof_phi, ess_tdof_p;
   spaces[0]->GetEssentialTrueDofs(*ess_bdr_marker[0], ess_tdof_j);
   spaces[1]->GetEssentialTrueDofs(*ess_bdr_marker[1], ess_tdof_phi);
   spaces[2]->GetEssentialTrueDofs(*ess_bdr_marker[2], ess_tdof_u);
   spaces[3]->GetEssentialTrueDofs(*ess_bdr_marker[3], ess_tdof_p);

   // Pin a pressure DoF.  This may not be necessary if we are also doing
   // OrthoSolver.
   ess_tdof_p.SetSize(1);
   ess_tdof_p[0] = 1.0;

   ParGridFunction j_gf(spaces[0]), phi_gf(spaces[1]), p_gf(spaces[3]);
   j_gf = 0.0; phi_gf = 0.0; u_gf = 0.0; p_gf = 0.0;

   // Project BCs onto grid functions.
   j_gf.ProjectBdrCoefficient(*currentD_DBC, *ess_bdr_marker[0]);
   phi_gf.ProjectBdrCoefficient(*electPot_DBC, *ess_bdr_marker[1]);
   u_gf.ProjectBdrCoefficient(*velocity_DBC, *ess_bdr_marker[2]);
   p_gf.ProjectBdrCoefficient(*pressure_DBC, *ess_bdr_marker[3]);

   ustar_gf = new ParGridFunction(spaces[2]);
   ustar_gf->ProjectBdrCoefficient(*velocity_DBC, *ess_bdr_marker[2]); // Project BC on first time-step.
   ustar_coef = std::make_unique<VectorGridFunctionCoefficient>(ustar_gf);

   //MeanZero(p_gf);

   // Mixed bilinear form for velocity and pressure coupling.
   bCoeff = new ConstantCoefficient(-1.0);
   //cout << "bCoeff value = " << bCoeff->constant << endl;
   fluids.b.AddDomainIntegrator(new VectorDivergenceIntegrator(*bCoeff));
   fluids.b.Assemble();
   fluids.b.Finalize();

   // Bilinear form for current density.
   mjCoeff = new ConstantCoefficient(kappa_val);
   //cout << "mjCoeff value = " << mjCoeff->constant << endl;
   magnetics.mj.AddDomainIntegrator(new VectorFEMassIntegrator(*mjCoeff));
   magnetics.mj.Assemble();
   magnetics.mj.Finalize();

   // Mixed bilinear form for current density and electric potential coupling.
   gCoeff = new ConstantCoefficient(-kappa_val);
   //cout << "gCoeff value = " << gCoeff->constant << endl;
   magnetics.g.AddDomainIntegrator(new MixedScalarDivergenceIntegrator(*gCoeff));
   magnetics.g.Assemble();
   magnetics.g.Finalize();

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
   coupling.k.AddDomainIntegrator(new VectorFEMassIntegrator(*kCoeff));
   coupling.k.Assemble();
   coupling.k.Finalize();


   // Integrator for (v, v').
   fluids.fu.AddDomainIntegrator(new VectorMassIntegrator(*fMassCoeff));
   // Integrator for A_AL(v, v').
   fluids.fu.AddDomainIntegrator(new VectorDiffusionIntegrator(*fReciprocalReCoeff));
   // Integrator for O(u_n; v, v').  ADD BOUNDARY TERM HERE.
   fluids.fu.AddDomainIntegrator(new VectorConvectionIntegrator(*ustar_coef,0.5));
   fluids.fu.AddDomainIntegrator(new ConservativeVectorConvectionIntegrator(*ustar_coef,-0.5));

   fluids.fu.Assemble();
   fluids.fu.Finalize();
   

   Vector Rj, Rphi, Ru, Rp, Xu_dummy, Xp_dummy, Ru_dummy;

   fluids.b.FormRectangularLinearSystem(ess_tdof_u, ess_tdof_p,
                               u_gf, rp, fluids.BMat_h,
                               Xu_dummy, Rp);

   magnetics.mj.FormLinearSystem(ess_tdof_j, j_gf, rj, magnetics.MjMat_h, Xu_dummy, Rj);

   magnetics.g.FormRectangularLinearSystem(ess_tdof_j, ess_tdof_phi,
                               j_gf, rphi, magnetics.GMat_h,
                               Xu_dummy, Rphi);

   coupling.k.FormRectangularLinearSystem(ess_tdof_j, ess_tdof_u,
                                j_gf, ru, coupling.KMat_h,
                                Ru_dummy, Xu_dummy);

   fluids.fu.FormLinearSystem(ess_tdof_u, u_gf, ru, fluids.FuMat_h, Xu_dummy, Ru);

   HypreParMatrix *BMat = fluids.BMat_h.As<HypreParMatrix>();
   HypreParMatrix *FuMat = fluids.FuMat_h.As<HypreParMatrix>();
   HypreParMatrix *MjMat = magnetics.MjMat_h.As<HypreParMatrix>();
   HypreParMatrix *GMat = magnetics.GMat_h.As<HypreParMatrix>();
   HypreParMatrix *KMat = coupling.KMat_h.As<HypreParMatrix>();

   smallPressureCoeff = new ConstantCoefficient(1e-12);
   fluids.smallPressure.AddDomainIntegrator(new MassIntegrator(*smallPressureCoeff));
   fluids.smallPressure.Assemble(); 
   fluids.smallPressure.Finalize();

   fluids.smallPressure.FormLinearSystem(ess_tdof_p, p_gf, rp, 
      fluids.smallPressureMat_h, Xp_dummy, Rp);
   HypreParMatrix *smallPressureMat = fluids.smallPressureMat_h.As<HypreParMatrix>();


   // Transpose of KMat with negative sign as KMat is negative and KtMat positive.
   HypreParMatrix *KtMat = KMat->Transpose();
   (*KtMat) *= -1.0;

   HypreParMatrix *GtMat = GMat->Transpose();
   HypreParMatrix *BtMat = BMat->Transpose();

   if (debug == 1)
   {
      BMat->Print("BMat.dat");
      BtMat->Print("BtMat.dat");
      MjMat->Print("MjMat.dat");
      GMat->Print("GMat.dat");
      GtMat->Print("GTMat.dat");
      KMat->Print("KMat.dat");
      KtMat->Print("KtMat.dat");
   }

   MFEM_VERIFY(BMat  != nullptr, "BMat null" );
   MFEM_VERIFY(BtMat != nullptr, "BtMat null");
   MFEM_VERIFY(MjMat != nullptr, "MjMat null");
   MFEM_VERIFY(GMat  != nullptr, "GMat null" );
   MFEM_VERIFY(GtMat != nullptr, "GTMat null");
   MFEM_VERIFY(KMat  != nullptr, "KMat null" );
   MFEM_VERIFY(KtMat != nullptr, "KtMat null");
   MFEM_VERIFY(FuMat != nullptr, "FuMat null");


   // Set up operator matrix for iterative solve (using FGMRES) or direct solve (using MUMPS).
   if (DIRECTSOLVE == 0)
   {
      A = std::make_unique<BlockOperator>(block_trueOffsets);
      dynamic_cast<BlockOperator*>(A.get())->SetBlock(0,0, MjMat);
      dynamic_cast<BlockOperator*>(A.get())->SetBlock(0,1, GtMat);
      dynamic_cast<BlockOperator*>(A.get())->SetBlock(1,0, GMat);
      // Set coupling (B^T and B) blocks.
      dynamic_cast<BlockOperator*>(A.get())->SetBlock(2,3, BtMat);
      dynamic_cast<BlockOperator*>(A.get())->SetBlock(3,2, BMat);
      // Set K blocks for coupling J and U.
      dynamic_cast<BlockOperator*>(A.get())->SetBlock(2,0, KMat);
      dynamic_cast<BlockOperator*>(A.get())->SetBlock(0,2, KtMat);
      dynamic_cast<BlockOperator*>(A.get())->SetBlock(2,2, fluids.FuMat_h.As<HypreParMatrix>());
      dynamic_cast<BlockOperator*>(A.get())->SetBlock(3,3, smallPressureMat);

      MFEM_VERIFY(A != nullptr, "A is null");
   }
   else
   {
      blocks = new Array2D<HypreParMatrix *>(4,4);
      for (int i = 0; i < 4; i++)
      {
         for (int j = 0; j < 4; j++) 
         {
            (*blocks)(i,j) = nullptr;
         }
      }
      
      (*blocks)(0,0) = MjMat;
      (*blocks)(0,1) = GtMat;
      (*blocks)(1,0) = GMat;
      // Set coupling (B^T and B) blocks.
      (*blocks)(2,3) = BtMat;
      (*blocks)(3,2) = BMat;
      // Set K blocks for coupling J and U.
      (*blocks)(2,0) = KMat;
      (*blocks)(0,2) = KtMat;
      (*blocks)(2,2) = fluids.FuMat_h.As<HypreParMatrix>();
      (*blocks)(3,3) = smallPressureMat;

      A = std::unique_ptr<HypreParMatrix>(HypreParMatrixFromBlocks(*blocks));

      if (debug == 1) dynamic_cast<HypreParMatrix*>(A.get())->Print("A.dat");
   }

   // Set RHS.
   RHS = new BlockVector(block_trueOffsets);
   RHS->GetBlock(0) = Rj;
   RHS->GetBlock(1) = Rphi;
   RHS->GetBlock(2) = Ru;
   RHS->GetBlock(3) = Rp; 

   
   //*****************************************************************************************************
   //**************************************** Preconditioner *********************************************
   //*****************************************************************************************************

   /// Define integrators for preconditioner.
   // Current density preconditioner.
   djCoeff = new ConstantCoefficient(kappa_val); // Including kappa here although not present in algorithm 4.1.
   //cout << "djCoeff value = " << djCoeff->constant << endl;
   //dj = new ParBilinearForm(spaces[0]);
   liprec.dj.AddDomainIntegrator(new VectorFEMassIntegrator(*djCoeff));  
   liprec.dj.AddDomainIntegrator(new DivDivIntegrator(*djCoeff));  
   liprec.dj.Assemble();
   liprec.dj.Finalize();

   // Electric potential preconditioner.
   mphiCoeff = new ConstantCoefficient(kappa_val); // Including kappa here although not present in algorithm 4.1.
   //mphiCoeff = new ConstantCoefficient(-kappa_val); // Including kappa here although not present in algorithm 4.1.
   //cout << "mphiCoeff value = " << mphiCoeff->constant << endl;
   //mphi = new ParBilinearForm(spaces[1]);
   liprec.mphi.AddDomainIntegrator(new MassIntegrator(*mphiCoeff));
   liprec.mphi.Assemble();
   liprec.mphi.Finalize();

   // Pressure preconditioner (part 1).
   mpCoeff = new ConstantCoefficient(1.0);
   //cout << "mpCoeff value = " << mpCoeff->constant << endl;
   //mp = new ParBilinearForm(spaces[3]);
   liprec.mp.AddDomainIntegrator(new MassIntegrator(*mpCoeff));
   liprec.mp.Assemble();
   liprec.mp.Finalize();

   // Pressure preconditioner (part 2).
   spCoeff = new ConstantCoefficient(1.0);
   //cout << "spCoeff value = " << spCoeff->constant << endl;
   //sp = new ParBilinearForm(spaces[3]);
   liprec.sp.AddDomainIntegrator(new DiffusionIntegrator(*spCoeff));
   liprec.sp.Assemble();
   liprec.sp.Finalize();

   // Bilinear form for velocity preconditioner.
   // Integrator for (v, v').
   liprec.fk.AddDomainIntegrator(new VectorMassIntegrator(*fMassCoeff));
   // Integrator for A_AL(v, v').
   liprec.fk.AddDomainIntegrator(new VectorDiffusionIntegrator(*fReciprocalReCoeff));
   // Integrator for O(u_n; v, v').
   liprec.fk.AddDomainIntegrator(new VectorConvectionIntegrator(*ustar_coef,0.5));
   liprec.fk.AddDomainIntegrator(new ConservativeVectorConvectionIntegrator(*ustar_coef,-0.5));
   liprec.fk.AddDomainIntegrator(new VectorMassIntegrator(*fkBxVBxVcoeff));
   liprec.fk.Assemble();
   liprec.fk.Finalize();

   
   liprec.dj.FormLinearSystem(  ess_tdof_j,   j_gf,   rj,   liprec.DjMat_h,   Xu_dummy, Rj);
   liprec.mphi.FormLinearSystem(ess_tdof_phi, phi_gf, rphi, liprec.MphiMat_h, Xu_dummy, Rphi);
   liprec.mp.FormLinearSystem(  ess_tdof_p,   p_gf,   rp,   liprec.MpMat_h,   Xu_dummy, Rp);
   liprec.sp.FormLinearSystem(  ess_tdof_p,   p_gf,   rp,   liprec.SpMat_h,   Xu_dummy, Rp);
   liprec.fk.FormLinearSystem(  ess_tdof_u,   u_gf,   ru,   liprec.FkMat_h,   Xu_dummy, Ru_dummy);

   HypreParMatrix *DjMat = liprec.DjMat_h.As<HypreParMatrix>();
   HypreParMatrix *MphiMat = liprec.MphiMat_h.As<HypreParMatrix>();
   HypreParMatrix *MpMat = liprec.MpMat_h.As<HypreParMatrix>();
   HypreParMatrix *SpMat = liprec.SpMat_h.As<HypreParMatrix>();
   HypreParMatrix *FkMat = liprec.FkMat_h.As<HypreParMatrix>();


   if (debug == 1)
   {
      DjMat->Print("DjMat.dat");
      MphiMat->Print("MphiMat.dat");
      MpMat->Print("MpMat.dat");
      SpMat->Print("SpMat.dat");
   }

   MFEM_VERIFY(MpMat  != nullptr, "MpMat null" );
   MFEM_VERIFY(SpMat != nullptr, "SpMat null");
   MFEM_VERIFY(MphiMat != nullptr, "MphiMat null");
   MFEM_VERIFY(DjMat  != nullptr, "DjMat null" );
   MFEM_VERIFY(FkMat  != nullptr, "FkMat null" );

   // Scaling Mp and Sp.
   // Compute max diagonal entry of Mp and Sp
   /*Vector MpDiag(MpMat->Height()), SpDiag(SpMat->Height());
   MpMat->GetDiag(MpDiag);
   SpMat->GetDiag(SpDiag);

   real_t MpMax = MpDiag.Max();
   real_t SpMax = SpDiag.Max();

   real_t sMp = 1.0 / MpMax;
   real_t sSp = 1.0 / SpMax;*/

   // Normalising pressure preconditioner (there's a better way to do this - currently just a test).
   /*mpCoeffNorm = new ConstantCoefficient(sMp);
   mpNorm = new ParBilinearForm(spaces[3]);
   mpNorm->AddDomainIntegrator(new MassIntegrator(*mpCoeffNorm));
   mpNorm->Assemble(); mpNorm->Finalize();

   MpMatNorm = new HypreParMatrix();
   mpNorm->FormLinearSystem(ess_tdof_p, p_gf, *rp, *MpMatNorm, Xu_dummy, Rp);

   spCoeffNorm = new ConstantCoefficient(sSp);
   spNorm = new ParBilinearForm(spaces[3]);
   spNorm->AddDomainIntegrator(new DiffusionIntegrator(*spCoeffNorm));
   spNorm->Assemble(); spNorm->Finalize();

   SpMatNorm = new HypreParMatrix();
   spNorm->FormLinearSystem(ess_tdof_p, p_gf, *rp, *SpMatNorm, Xu_dummy, Rp);

   if (debug == 1)
   {
      MpMatNorm->Print("MpMatNorm.dat");
      SpMatNorm->Print("SpMatNorm.dat");
   }*/

   P = new LiPreconditioner(spaces, block_trueOffsets, dt);

   // SetPressurePreconditioner is currently causing a memory issue.
   P->SetPressurePreconditioner(MpMat, SpMat);
   P->SetElectricPotentialPreconditioner(MphiMat);
   P->SetCurrentDensityPreconditioner(DjMat, GtMat, KtMat);

   P->SetVelocityPreconditioner(FkMat, BtMat);

   //*****************************************************************************************************
   //**************************************** Preconditioner *********************************************
   //*****************************************************************************************************

   lmmhd_solver->SetOperator(*A);

   if (DIRECTSOLVE == 0) dynamic_cast<FGMRESSolver*>(lmmhd_solver.get())->SetPreconditioner(*P);

}

void LmmhdOperator::Update(const Vector &X)
{
   // Bilinear form for the velocity.
   fluids.fu.Update();
   fluids.fu.Assemble();
   fluids.fu.Finalize();

   Vector Ru, Xu_dummy, Ru_dummy;
   fluids.FuMat_h.Clear();
   fluids.fu.FormLinearSystem(ess_tdof_u, u_gf, ru, fluids.FuMat_h, Xu_dummy, Ru);
   //FuMat = fluids.FuMat_h.As<HypreParMatrix>();

   // Update operator matrix for iterative solve (using FGMRES) or direct solve (using MUMPS).
   if (DIRECTSOLVE == 0)
   {
      dynamic_cast<BlockOperator*>(A.get())->SetBlock(2,2, fluids.FuMat_h.As<HypreParMatrix>());
   }
   else
   {
      (*blocks)(2,2) = fluids.FuMat_h.As<HypreParMatrix>();
      A = std::unique_ptr<HypreParMatrix>(HypreParMatrixFromBlocks(*blocks));
   }

   // Update other residuals?
   RHS->GetBlock(2) = Ru;


   // Preconditioner.
   liprec.fk.Update();
   liprec.fk.Assemble();
   liprec.fk.Finalize();

   liprec.FkMat_h.Clear();
   liprec.fk.FormLinearSystem(ess_tdof_u, u_gf, ru, liprec.FkMat_h, Xu_dummy, Ru_dummy);

   P->UpdateVelocityPreconditioner(liprec.FkMat_h.As<HypreParMatrix>());
   
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

LmmhdOperator::~LmmhdOperator() {

   delete prec_ortho_solver;

   delete currentD_DBC;
   delete electPot_DBC;
   delete pressure_DBC;

   delete zeroCoeff;
   delete vectorZeroCoeff;

   delete velocity_DBC;

   delete B;

   delete fkBxVBxVcoeff;

   delete fMassCoeff;
   delete fReciprocalReCoeff;

   delete ustar_gf;

   delete bCoeff;

   delete mjCoeff;

   delete gCoeff;

   delete kCoeffVec;
   delete kCoeff;

   delete smallPressureCoeff;

   delete RHS;

   delete djCoeff;
   delete mphiCoeff;
   delete mpCoeff;
   delete spCoeff;

   delete P;

   delete blocks;

   delete BtMat;
   delete GtMat;
   delete KtMat;

}