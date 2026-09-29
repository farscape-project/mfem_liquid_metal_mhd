#include "LmmhdOperator.hpp"

LmmhdOperator::LmmhdOperator(Array<ParFiniteElementSpace *> &fes,
                            Array<Array<int> *> &ess_bdr,
                            Array<int> &offsets,
                            int dim_,
                            real_t dt_,
                            int debug_,
                            Logger &logger_)
   : TimeDependentOperator(OperatorSize(fes)),
      block_trueOffsets(offsets),
      magnetics(fes[0],fes[1]),
      fluids(fes[2],fes[3]),
      coupling(fes[0],fes[2]),
      liprec(fes[0],fes[1],fes[2],fes[3]),
      rj(fes[0]),
      rphi(fes[1]),
      ru(fes[2]),
      rp(fes[3]),
      lmmhd_solver(),
      dim(dim_),
      dt(dt_),
      debug(debug_),
      logger(logger_),
      potential_mean_remover(*fes[1]),
      pressure_mean_remover(*fes[3])
   {
   // Sets up the linear system for the coupled MHD solve:
   //
   //       [  Mj    G^T    K^T       0   ] [ xj    ]   [ rj   ]
   //       [  G     0      0         0   ] [ xphi  ]   [ rphi ]
   //       [ -K     0      Fk       B^T  ] [ xubar ] = [ ru   ]
   //       [  0     0      B         0   ] [ xp    ]   [ rp   ]
   //
   // where:
   //   - Mj    : current-density bilinear form: (d,d')
   //   - G     : coupling between current and electric potential: -(div d, phi)
   //   - K     : coupling between current and velocity: (d, B x v')
   //   - Fk    : velocity bilinear form: 2/Tau (v, v') + O(u*_n; v, v') + A_AL(v, v')
   //   - B     : coupling between velocity and pressure: -(div v, q)

   lmmhd_solver = new FGMRESSolver(MPI_COMM_WORLD);
   lmmhd_solver->SetRelTol(1e-8);
   lmmhd_solver->SetAbsTol(1e-12);
   lmmhd_solver->SetMaxIter(300);
   lmmhd_solver->SetPrintLevel(1);
   lmmhd_solver->SetKDim(50);
   lmmhd_solver->iterative_mode = false;

   // Add monitoring for output to log file.
   fgmres_monitor = new FGMRESLogMonitor(logger);
   lmmhd_solver->SetMonitor(*fgmres_monitor);

   //prec_ortho_solver = new OrthoSolver(MPI_COMM_WORLD);

   fes.Copy(spaces);
   ess_bdr.Copy(ess_bdr_marker);

   X = new BlockVector(block_trueOffsets);
   Xn_1 = new BlockVector(block_trueOffsets);
   RHS = new BlockVector(block_trueOffsets);
   *Xn_1 = 0.0;

   A = new BlockOperator(block_trueOffsets);

   // Set Dirichlet boundary condition functions (not time-dependent).
   currentD_DBC = new VectorFunctionCoefficient(dim, currentD_dbc_vec_func);
   electPot_DBC = new FunctionCoefficient(electPot_dbc);
   pressure_DBC = new FunctionCoefficient(pressure_dbc);

   zeroCoeff = new ConstantCoefficient(0.0);
   oneCoeff = new ConstantCoefficient(1.0);
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

   massCoeffValue = 2.0 / dt;
   fMassCoeff = new ConstantCoefficient(massCoeffValue);
   fReciprocalReCoeff = new ConstantCoefficient(reciprocal_Re);
   alphaCoeff = new ConstantCoefficient(alpha);


   /// Set up RHS linear forms.
   // Set up rhs for velocity solve.
   ru.AddDomainIntegrator(new VectorDomainLFIntegrator(*vectorZeroCoeff));
   ru.Assemble();

   // Set up rhs for pressure solve.
   // NOTE: these must assemble the *member* linear forms.  Declaring local
   // ParLinearForms here shadowed the members, which were then used
   // unassembled (MFEM's Vector(int) does not zero its entries).
   rp.AddDomainIntegrator(new DomainLFIntegrator(*zeroCoeff));
   rp.Assemble();

   // Set up rhs for current density solve.
   rj.AddDomainIntegrator(new VectorFEDomainLFIntegrator(*vectorZeroCoeff));
   rj.Assemble();

   // Set up rhs for electric potential solve.
   rphi.AddDomainIntegrator(new DomainLFIntegrator(*zeroCoeff));
   rphi.Assemble();

   // Dirichlet boundary conditions.
   //Array<int> ess_tdof_j, ess_tdof_phi, ess_tdof_p;
   spaces[0]->GetEssentialTrueDofs(*ess_bdr_marker[0], ess_tdof_j);
   spaces[1]->GetEssentialTrueDofs(*ess_bdr_marker[1], ess_tdof_phi);
   spaces[2]->GetEssentialTrueDofs(*ess_bdr_marker[2], ess_tdof_u);
   spaces[3]->GetEssentialTrueDofs(*ess_bdr_marker[3], ess_tdof_p);

   // Pin a pressure DoF.
   //ess_tdof_p.SetSize(1);
   //ess_tdof_p[0] = 1.0;

   // Set up grid functions.
   j_gf.SetSpace(spaces[0]);
   phi_gf.SetSpace(spaces[1]);
   u_gf.SetSpace(spaces[2]);
   p_gf.SetSpace(spaces[3]);
   j_gf = 0.0; phi_gf = 0.0; u_gf = 0.0; p_gf = 0.0;

   // Midpoint grid function.
   ubar_gf.SetSpace(spaces[2]);
   ubar_gf = 0.0;

   // Set up grid function history.
   u_gf_n_1.SetSpace(spaces[2]);
   u_gf_n_2.SetSpace(spaces[2]);
   u_gf_n_1 = 0.0;
   u_gf_n_2 = 0.0;

   j_gf_n_1.SetSpace(spaces[0]);
   phi_gf_n_1.SetSpace(spaces[1]);
   p_gf_n_1.SetSpace(spaces[3]);
   j_gf_n_1 = 0.0;
   phi_gf_n_1 = 0.0;
   p_gf_n_1 = 0.0;

   // Coefficient for u_n_1.
   velocity_n_1_Coeff = new VectorGridFunctionCoefficient(&u_gf_n_1);

   // ubar boundary condition:
   // ubar_n = 0.5 * (g + u_n_1).
   velocity_bar_DBC = new VectorSumCoefficient(*velocity_DBC, *velocity_n_1_Coeff, 0.5, 0.5);
   //(*velocity_bar_DBC) *= 0.5;

   // Project BCs onto grid functions.
   j_gf.ProjectBdrCoefficientNormal(*currentD_DBC, *ess_bdr_marker[0]);
   phi_gf.ProjectBdrCoefficient(*electPot_DBC, *ess_bdr_marker[1]);
   u_gf.ProjectBdrCoefficient(*velocity_DBC, *ess_bdr_marker[2]);
   //ubar_gf.ProjectBdrCoefficient(*velocity_DBC, *ess_bdr_marker[2]);
   ubar_gf.ProjectBdrCoefficient(*velocity_bar_DBC, *ess_bdr_marker[2]);
   p_gf.ProjectBdrCoefficient(*pressure_DBC, *ess_bdr_marker[3]);

   ustar_gf = new ParGridFunction(spaces[2]);
   ustar_gf->ProjectBdrCoefficient(*velocity_DBC, *ess_bdr_marker[2]); // Project BC on first time-step.
   ustar_coef = std::make_unique<VectorGridFunctionCoefficient>(ustar_gf);

   // Mixed bilinear form for velocity and pressure coupling.
   bCoeff = new ConstantCoefficient(-1.0);
   //cout << "bCoeff value = " << bCoeff->constant << endl;
   fluids.b.AddDomainIntegrator(new VectorDivergenceIntegrator(*bCoeff));
   fluids.b.Assemble();
   fluids.b.Finalize();

   // Bilinear form for current density.
   mjCoeff = new ConstantCoefficient(1.0);
   //cout << "mjCoeff value = " << mjCoeff->constant << endl;
   magnetics.mj.AddDomainIntegrator(new VectorFEMassIntegrator(*mjCoeff));
   magnetics.mj.Assemble();
   magnetics.mj.Finalize();

   // Mixed bilinear form for current density and electric potential coupling.
   gCoeff = new ConstantCoefficient(-1.0);
   //cout << "gCoeff value = " << gCoeff->constant << endl;
   magnetics.g.AddDomainIntegrator(new MixedScalarDivergenceIntegrator(*gCoeff));
   magnetics.g.Assemble();
   magnetics.g.Finalize();

   // Mixed bilinear form for current density and velocity coupling.
   // VectorFEMassIntegrator(lambda) applies (lambda d, v').  In order to 
   // apply (d, B x v') we rearrange the identity to (-B x d, v') using the 
   // identities a . (b x c) = c . (a x b) and (a x b) = - (b x a).
   kCoeffVec = new Vector(dim);
   *kCoeffVec = 0.0;
   *kCoeffVec -= *B;
   //std::cout << "kCoeffVec: " << (*kCoeffVec)(0) << ", " << (*kCoeffVec)(1) << ", " << (*kCoeffVec)(2) << std::endl;
   kCoeff = new CrossProductMatrixCoefficient(*kCoeffVec);
   coupling.k.AddDomainIntegrator(new VectorFEMassIntegrator(*kCoeff));
   coupling.k.Assemble();
   coupling.k.Finalize();

   // Keep an un-eliminated copy of K^T for lifting the velocity Dirichlet
   // data into the current-density (Ohm's law) equation.  The matrix returned
   // by FormRectangularLinearSystem has the essential velocity (test) rows
   // zeroed, so its transpose cannot be used for this lifting.  This must be
   // done before the first FormRectangularLinearSystem call, which deletes the
   // local sparse matrix.
   {
      HypreParMatrix *KFull = coupling.k.ParallelAssemble();
      KtFullMat = KFull->Transpose();
      delete KFull;
   }


   // Integrator for (v, v').
   fluids.fu.AddDomainIntegrator(new VectorMassIntegrator(*fMassCoeff));
   // Integrators for A_AL(v, v') = 1/Re (grad v, grad v') + alpha (div v, div v').
   // The grad-div (augmented Lagrangian) term must be present in both the
   // system and the preconditioner: the pressure Schur-complement
   // approximation in LiPreconditioner uses alpha1 = alpha + 1/Re.
   // ElasticityIntegrator(lambda, mu) with mu = 0 gives lambda (div v, div v').
   fluids.fu.AddDomainIntegrator(new VectorDiffusionIntegrator(*fReciprocalReCoeff));
   if (alpha != 0.0)
   {
      fluids.fu.AddDomainIntegrator(new ElasticityIntegrator(*alphaCoeff, *zeroCoeff));
   }
   // Integrator for the skew-symmetric convection form
   //    O(u*; v, v') = 1/2 (u*.grad v, v') - 1/2 (u*.grad v', v).
   // ConservativeVectorConvectionIntegrator(q, a) represents -a (v, q.grad v'),
   // so a = +0.5 gives the required -1/2 (u*.grad v', v).  (Passing -0.5 here
   // produces the symmetric part of convection, which integrates to
   // -1/2 ((div u*) v, v') and removes transport from the momentum equation.)
   // TODO: outflow boundary term for open (duct) configurations.
   fluids.fu.AddDomainIntegrator(new VectorConvectionIntegrator(*ustar_coef, 0.5));
   fluids.fu.AddDomainIntegrator(new ConservativeVectorConvectionIntegrator(*ustar_coef, 0.5));

   smallPressureCoeff = new ConstantCoefficient(1e-12);
   fluids.smallPressure.AddDomainIntegrator(new MassIntegrator(*smallPressureCoeff));
   fluids.smallPressure.Assemble(); 
   fluids.smallPressure.Finalize();

   rhs_mu = new ParBilinearForm(spaces[2]);
   rhs_mu->AddDomainIntegrator(new VectorMassIntegrator(*fMassCoeff));
   rhs_mu->Assemble();
   rhs_mu->Finalize();


   //*****************************************************************************************************//
   //**************************************** Preconditioner *********************************************//
   //*****************************************************************************************************//

   /// Define integrators for preconditioner.
   // Current density preconditioner.
   djCoeff = new ConstantCoefficient(1.0);
   liprec.dj.AddDomainIntegrator(new VectorFEMassIntegrator(*djCoeff));  
   liprec.dj.AddDomainIntegrator(new DivDivIntegrator(*djCoeff));  
   liprec.dj.Assemble();
   liprec.dj.Finalize();

   // Electric potential preconditioner.
   mphiCoeff = new ConstantCoefficient(1.0);
   liprec.mphi.AddDomainIntegrator(new MassIntegrator(*mphiCoeff));
   liprec.mphi.Assemble();
   liprec.mphi.Finalize();

   // Pressure preconditioner (part 1).
   mpCoeff = new ConstantCoefficient(1.0);
   liprec.mp.AddDomainIntegrator(new MassIntegrator(*mpCoeff));
   liprec.mp.Assemble();
   liprec.mp.Finalize();

   // Pressure preconditioner (part 2).
   spCoeff = new ConstantCoefficient(1.0);
   liprec.sp.AddDomainIntegrator(new DiffusionIntegrator(*spCoeff));
   liprec.sp.Assemble();
   liprec.sp.Finalize();

   // Bilinear form for velocity preconditioner.
   // Integrator for (v, v').
   liprec.fk.AddDomainIntegrator(new VectorMassIntegrator(*fMassCoeff));
   // Integrators for A_AL(v, v').
   liprec.fk.AddDomainIntegrator(new VectorDiffusionIntegrator(*fReciprocalReCoeff));
   if (alpha != 0.0)
   {
      liprec.fk.AddDomainIntegrator(new ElasticityIntegrator(*alphaCoeff, *zeroCoeff));
   }
   // Integrator for O(u_n; v, v').
   liprec.fk.AddDomainIntegrator(new VectorConvectionIntegrator(*ustar_coef, 0.5));
   liprec.fk.AddDomainIntegrator(new ConservativeVectorConvectionIntegrator(*ustar_coef, 0.5));
   liprec.fk.AddDomainIntegrator(new VectorMassIntegrator(*fkBxVBxVcoeff));

   P = new LiPreconditioner(spaces, block_trueOffsets, dt, logger);

   //*****************************************************************************************************//
   //**************************************** Preconditioner *********************************************//
   //*****************************************************************************************************//

   RemoveMeanProjector potential_mean_remover(*spaces[1]);
   RemoveMeanProjector pressure_mean_remover(*spaces[3]);

}

int LmmhdOperator::OperatorSize(const Array<ParFiniteElementSpace *> fes)
{
   int op_size = 0;
   for(int i = 0; i < fes.Size(); i++)
   {
      op_size += fes[i]->GetTrueVSize();
   }

   return op_size;
};

void LmmhdOperator::UpdateUStar(int step)
{
   if (step == 0)
   {
      *ustar_gf = u_gf;
   }
   else if (step == 1)
   {
      *ustar_gf = u_gf_n_1;
   }
   else
   {
      *ustar_gf = u_gf_n_1;
      *ustar_gf *= 3.0;
      ustar_gf->Add(-1.0, u_gf_n_2);
      *ustar_gf *= 0.5;
   }

   //*ustar_gf = 0.0;
   *ustar_coef = ustar_gf;
}

void LmmhdOperator::UpdateIntegrators()
{
   fluids.fu.Update();
   fluids.fu.Assemble();
   fluids.fu.Finalize();

   liprec.fk.Update();
   liprec.fk.Assemble();
   liprec.fk.Finalize();
}

void LmmhdOperator::RemoveMeans()
{
   // Remove mean from pressure and potential.  Only do this if nullspace
   // needs removing from the problem (i.e. all Neumann BCs).
   potential_mean_remover.RemoveMean(X->GetBlock(1));
   pressure_mean_remover.RemoveMean(X->GetBlock(3));
}

void LmmhdOperator::SetGridFunctionsFromTrueDofs()
{
   j_gf.SetFromTrueDofs(X->GetBlock(0));
   phi_gf.SetFromTrueDofs(X->GetBlock(1));
   ubar_gf.SetFromTrueDofs(X->GetBlock(2));
   p_gf.SetFromTrueDofs(X->GetBlock(3));
}

void LmmhdOperator::ReconstructPhysicalVelocityFromUBar(int step)
{
   // Reconstruct physical velocity:
   if (step == 0)
   {
      u_gf = ubar_gf;
   }
   else
   {
      // u_n = 2*u_bar_n - u_{n-1}
      u_gf = ubar_gf;
      u_gf *= 2.0;
      u_gf -= u_gf_n_1;
   }
}

void LmmhdOperator::CalcNorms()
{
      // Calculate relative L2-norm of grid functions between this and previous time-step.
      real_t cd_rel_l2 = rel_L2_norm(j_gf, j_gf_n_1, spaces[0]);
      if (Mpi::Root()) {std::cout << "Relative L2 Norm for current density: " << cd_rel_l2 << std::endl;}
      logger << "Relative L2 Norm for current density: " << cd_rel_l2 << std::endl;

      real_t elp_rel_l2 = rel_L2_norm(phi_gf, phi_gf_n_1, spaces[1]);
      if (Mpi::Root()) {std::cout << "Relative L2 Norm for electric potential: " << elp_rel_l2 << std::endl;}
      logger << "Relative L2 Norm for electric potential: " << elp_rel_l2 << std::endl;

      real_t vel_rel_l2 = rel_L2_norm(u_gf, u_gf_n_1, spaces[2]);
      if (Mpi::Root()) {std::cout << "Relative L2 Norm for velocity: " << vel_rel_l2 << std::endl;}
      logger << "Relative L2 Norm for velocity: " << vel_rel_l2 << std::endl;

      real_t pres_rel_l2 = rel_L2_norm(p_gf, p_gf_n_1, spaces[3]);
      if (Mpi::Root()) {std::cout << "Relative L2 Norm for pressure: " << pres_rel_l2 << std::endl;}
      logger << "Relative L2 Norm for pressure: " << pres_rel_l2 << std::endl;
}

void LmmhdOperator::UpdateHistory()
{
   u_gf_n_2 = u_gf_n_1;
   u_gf_n_1 = u_gf;

   j_gf_n_1 = j_gf;
   phi_gf_n_1 = phi_gf;
   p_gf_n_1 = p_gf;
}

void LmmhdOperator::SetBCs()
{
   // Project BCs onto grid functions.
   j_gf.ProjectBdrCoefficientNormal(*currentD_DBC, *ess_bdr_marker[0]);
   phi_gf.ProjectBdrCoefficient(*electPot_DBC, *ess_bdr_marker[1]);
   u_gf.ProjectBdrCoefficient(*velocity_DBC, *ess_bdr_marker[2]);
   //ubar_gf.ProjectBdrCoefficient(*velocity_DBC, *ess_bdr_marker[2]);
   ubar_gf.ProjectBdrCoefficient(*velocity_bar_DBC, *ess_bdr_marker[2]);
   p_gf.ProjectBdrCoefficient(*pressure_DBC, *ess_bdr_marker[3]);
}

void LmmhdOperator::FormASystem()
{
   // Zero RHS before setting values.
   *RHS = 0.0;

   Vector Rp, Rphi, Ru, X_dummy;
   Vector aux_x, aux_rhs;

   magnetics.mj.FormLinearSystem(ess_tdof_j,
                                 j_gf,
                                 rj,
                                 magnetics.MjMat_h,
                                 X->GetBlock(0),
                                 RHS->GetBlock(0),
                                 true);

   // The mixed forms are only used to lift essential trial data into the
   // off-diagonal blocks, so they are given a zero test-space vector.  Passing
   // the diagonal blocks' linear forms here (as before) counted those forms
   // twice in the velocity and pressure rows.
   Vector zero_phi(spaces[1]->GetVSize()); zero_phi = 0.0;
   Vector zero_u(spaces[2]->GetVSize());   zero_u = 0.0;
   Vector zero_p(spaces[3]->GetVSize());   zero_p = 0.0;

   // Block (1,0) is G: lifting gives -G j_bc.
   magnetics.g.FormRectangularLinearSystem(ess_tdof_j,
                                             ess_tdof_phi,
                                             j_gf,
                                             zero_phi,
                                             magnetics.GMat_h,
                                             X_dummy,
                                             Rphi);
   RHS->GetBlock(1) += Rphi;

   fluids.fu.FormLinearSystem(ess_tdof_u,
                              ubar_gf,
                              ru,
                              fluids.FuMat_h,
                              X->GetBlock(2),
                              RHS->GetBlock(2),
                              true);

   // Block (2,0) is -kappa K: Ru = -K j_bc, so the lifting is -kappa * Ru.
   coupling.k.FormRectangularLinearSystem(ess_tdof_j,
                                          ess_tdof_u,
                                          j_gf,
                                          zero_u,
                                          coupling.KMat_h,
                                          X_dummy,
                                          Ru);
   RHS->GetBlock(2).Add(-kappa_val, Ru);

   fluids.smallPressure.FormLinearSystem(ess_tdof_p,
                                          p_gf,
                                          rp,
                                          fluids.smallPressureMat_h,
                                          X->GetBlock(3),
                                          RHS->GetBlock(3),
                                          true);

   // Block (3,2) is B: lifting gives -B ubar_bc.
   fluids.b.FormRectangularLinearSystem(ess_tdof_u,
                                          ess_tdof_p,
                                          ubar_gf,
                                          zero_p,
                                          fluids.BMat_h,
                                          X_dummy,
                                          Rp);
   RHS->GetBlock(3) += Rp;

   // Block (0,2) is K^T: lift the velocity Dirichlet data into Ohm's law
   // using the un-eliminated K^T, then restore the essential current rows.
   {
      Vector u_bc(spaces[2]->GetTrueVSize());
      u_bc = 0.0;
      const Vector &Xu = X->GetBlock(2);
      for (int i = 0; i < ess_tdof_u.Size(); i++)
      {
         u_bc(ess_tdof_u[i]) = Xu(ess_tdof_u[i]);
      }
      Vector Kt_bc(spaces[0]->GetTrueVSize());
      KtFullMat->Mult(u_bc, Kt_bc);
      Vector &Rj = RHS->GetBlock(0);
      Rj -= Kt_bc;
      const Vector &Xj = X->GetBlock(0);
      for (int i = 0; i < ess_tdof_j.Size(); i++)
      {
         Rj(ess_tdof_j[i]) = Xj(ess_tdof_j[i]);
      }
   }
}

void LmmhdOperator::FormPSystem()
{
  liprec.dj.FormSystemMatrix(ess_tdof_j, liprec.DjMat_h);
  liprec.mphi.FormSystemMatrix(ess_tdof_phi, liprec.MphiMat_h);
  liprec.mp.FormSystemMatrix(ess_tdof_p, liprec.MpMat_h);
  liprec.sp.FormSystemMatrix(ess_tdof_p, liprec.SpMat_h);
  liprec.fk.FormSystemMatrix(ess_tdof_u, liprec.FkMat_h);
}

void LmmhdOperator::Step(real_t &time, real_t dt)
{

   SetBCs();

   FormASystem(); // FormLinearSystem and FormRectangularLinearSystem calls for entries of A matrix.

   // Build A matrix.
   HypreParMatrix *BMat = fluids.BMat_h.As<HypreParMatrix>();
   HypreParMatrix *FuMat = fluids.FuMat_h.As<HypreParMatrix>();
   HypreParMatrix *MjMat = magnetics.MjMat_h.As<HypreParMatrix>();
   HypreParMatrix *GMat = magnetics.GMat_h.As<HypreParMatrix>();
   HypreParMatrix *KMat = coupling.KMat_h.As<HypreParMatrix>();
   HypreParMatrix *smallPressureMat = fluids.smallPressureMat_h.As<HypreParMatrix>();

   // Transposes.
   HypreParMatrix *GtMat = GMat->Transpose();
   HypreParMatrix *BtMat = BMat->Transpose();
   HypreParMatrix *KtMat = KMat->Transpose();

   HypreParMatrix *BMat_A = new HypreParMatrix(*BMat);
   HypreParMatrix *BtMat_A = new HypreParMatrix(*BtMat);
   HypreParMatrix *FuMat_A = new HypreParMatrix(*FuMat);
   HypreParMatrix *MjMat_A = new HypreParMatrix(*MjMat);
   HypreParMatrix *GMat_A = new HypreParMatrix(*GMat);
   HypreParMatrix *GtMat_A = new HypreParMatrix(*GtMat);
   HypreParMatrix *KMat_A = new HypreParMatrix(*KMat);
   HypreParMatrix *KtMat_A = new HypreParMatrix(*KtMat);
   HypreParMatrix *smallPressureMat_A = new HypreParMatrix(*smallPressureMat);

   // Apply kappa and negative signs in A matrix.
   //(*MjMat_A) *= kappa_val;
   //(*GMat_A) *= kappa_val;
   //(*GtMat_A) *= kappa_val;
   (*KMat_A) *= -1.0;
   (*KMat_A) *= kappa_val;   
   //(*KtMat_A) *= kappa_val;


   // (The velocity Dirichlet lifting into the current equation is done in
   // FormASystem() with the un-eliminated K^T.)


   // Calculate un_1 term for rhs of velocity equation.
   HypreParMatrix *MuMat;
   MuMat = rhs_mu->ParallelAssemble();

   HypreParVector u_old(spaces[2]);
   u_gf_n_1.GetTrueDofs(u_old);

   HypreParVector Mu_u_old(spaces[2]);
   MuMat->Mult(u_old, Mu_u_old);
   RHS->GetBlock(2).Add(1.0, Mu_u_old);


   A->SetBlock(0,0, MjMat_A);
   A->SetBlock(0,1, GtMat_A);
   A->SetBlock(1,0, GMat_A);
   // Set coupling (B^T and B) blocks.
   A->SetBlock(2,3, BtMat_A);
   A->SetBlock(3,2, BMat_A);
   // Set K blocks for coupling J and U.
   A->SetBlock(2,0, KMat_A);
   A->SetBlock(0,2, KtMat_A);
   A->SetBlock(2,2, FuMat_A);
   A->SetBlock(3,3, smallPressureMat_A);

   lmmhd_solver->SetOperator(*A);

   FormPSystem(); // FormLinearSystem and FormRectangularLinearSystem calls for preconditioner.

   // Build P matrices.
   HypreParMatrix *DjMat = liprec.DjMat_h.As<HypreParMatrix>();
   HypreParMatrix *MphiMat = liprec.MphiMat_h.As<HypreParMatrix>();
   HypreParMatrix *MpMat = liprec.MpMat_h.As<HypreParMatrix>();
   HypreParMatrix *SpMat = liprec.SpMat_h.As<HypreParMatrix>();
   HypreParMatrix *FkMat = liprec.FkMat_h.As<HypreParMatrix>();

   HypreParMatrix *GtMat_P = new HypreParMatrix(*GtMat);
   HypreParMatrix *KtMat_P = new HypreParMatrix(*KtMat);

   // Propagate matrices through to preconditioner.
   P->SetPressurePreconditioner(MpMat, SpMat);
   P->SetElectricPotentialPreconditioner(MphiMat);
   P->SetCurrentDensityPreconditioner(DjMat, GtMat_P, KtMat_P);
   P->SetVelocityPreconditioner(FkMat, BtMat);
   P->UpdateVelocityPreconditioner(FkMat);

   lmmhd_solver->SetPreconditioner(*P);

   lmmhd_solver->Mult(*RHS, *X);


   // Calculating residual only.
   Vector residual(*RHS);
   A->Mult(*X, residual);
   residual.Neg();
   residual += *RHS;

   real_t norm = residual.Norml2();
   real_t local_sq = residual * residual; // dot product
   real_t global_sq;
   MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   real_t global_norm = std::sqrt(global_sq);
   if (Mpi::Root()) std::cout << "global_norm = " << global_norm << std::endl;


   delete BMat_A;
   delete BtMat_A;
   delete FuMat_A;
   delete MjMat_A;
   delete GMat_A;
   delete GtMat_A;
   delete KMat_A;
   delete KtMat_A;
   delete smallPressureMat_A;
   delete GtMat_P;
   delete KtMat_P;
}

LmmhdOperator::~LmmhdOperator() {

   delete prec_ortho_solver;

   delete currentD_DBC;
   delete electPot_DBC;
   delete pressure_DBC;

   delete zeroCoeff;
   delete oneCoeff;
   delete vectorZeroCoeff;

   delete velocity_DBC;

   delete B;

   delete fkBxVBxVcoeff;

   delete fMassCoeff;
   delete fReciprocalReCoeff;
   delete alphaCoeff;

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
   delete KtFullMat;
}
