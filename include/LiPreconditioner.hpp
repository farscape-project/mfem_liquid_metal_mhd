#pragma once

#include "mfem.hpp"
#include "tools.hpp"
#include "parameters.hpp"

// Solves algorithm 4.1 from Li et al. 2019, where the matrix P acts as the 
// preconditioner s.t. Py = r where P is given by
//
//       [  k Dj    2 k G^T    2 k K^T       0   ]
//       [   0      -k Mphi       0          0   ]
//       [   0         0          Fk        B^T  ]
//       [   0         0          0       -Lp^-1 ]
//
// where:
//   - Dj    : (d,d') + (div d, div d')
//   - G     : -(div d, phi)
//   - K     : (d, B x v')
//   - Mphi  : (phi, phi')
//   - Fk    : velocity bilinear form: 2/Tau (v, v') + O(u*_n; v, v') + A_AL(v, v')
//   - B     : coupling between velocity and pressure: -(div v, q)
//   - Lp    : alpha_1 Mp^-1 + 2/tau Sp^-1 where
//   - Mp    : (q, q')
//   - Sp    : (grad q, grad q')
//
//
// Algorithm 4.1 is presented in Li et al. 2019 and is given by
//
//  1. Compute y_p = -Lp r_p = - alpha_1 xi - (2/tau) eta [note: this 2/tau is missing from Li et al.]
//      - solve Mp xi = r_p by 10 iterations of CG solver with diagonal preconditioner,
//      - solve Sp eta = r_p by 2 iterations of algebraic multigrid solver.
//
//  2. Solve Mphi y_phi = -r_phi by 10 iterations of CG solver with diagonal preconditioner.
//
//  3. Solve Fk y_u = r_u - B^T y_p by GMRES solver with additive Schwarz preconditioner.
//     The tolerance for relative residuals is set to 10^-3.
//     [here: GMRES + systems BoomerAMG]
//
//  4. Solve Dj y_j = r_j - 2 G^T y_phi - 2 K^T y_u by 5 iterations of CG solver with 
//     the HX preconditioner.
//     [here: MUMPS by default, or CG + HypreADS with use_ads_for_currentdensity = true]
//
// Scaling: the paper writes the J and phi rows of the system (and of P)
// multiplied by kappa.  LmmhdOperator assembles those rows unscaled (Mj, G^T,
// K^T and G) with -kappa K in the momentum row.  Left-multiplying both the
// system and the preconditioner by the same block-diagonal scaling leaves
// P^{-1} A unchanged, so the unscaled P below is consistent with that system.
//


class LiPreconditioner : public mfem::Solver
{
protected:
    mfem::Array<mfem::ParFiniteElementSpace *> spaces;
    int nBlocks;
    mfem::Array<int> offsets;
    mfem::real_t dt;
    mfem::real_t theta;
    mfem::real_t alpha1;

    // Pressure preconditioner solvers.
    mfem::CGSolver *MpSolver = nullptr;
    mfem::HypreSmoother *MpPrec = nullptr;
    mfem::HypreBoomerAMG *SpSolver = nullptr;
    mfem::OrthoSolver *SpOrthoSolver = nullptr;

    // Electric potential preconditioner solvers.
    mfem::CGSolver *MphiSolver = nullptr;
    mfem::HypreSmoother *MphiPrec = nullptr;

    // Velocity preconditioner solvers (rebuilt every time step).
    mfem::GMRESSolver *FkSolver = nullptr;
    mfem::HypreBoomerAMG *FkPrec = nullptr;
    mfem::HypreParMatrix *Bt = nullptr;   // not owned

    // Current density preconditioner solvers.
    mfem::Solver *DjSolver = nullptr;     // MUMPSSolver or CGSolver
    mfem::HypreADS *DjPrec = nullptr;     // only used with the CG + ADS option
    bool dj_use_ads;

    mfem::HypreParMatrix *Gt = nullptr;   // not owned
    mfem::HypreParMatrix *Kt = nullptr;   // not owned

    Logger &logger;

    // Per-application norm logging (collective; off by default because it
    // costs several global reductions per preconditioner application).
    bool verbose = false;

    mfem::real_t GNorm(const mfem::Vector &v) const { return ParNormlp(v, 2.0, MPI_COMM_WORLD); }

public:
    LiPreconditioner(mfem::Array<mfem::ParFiniteElementSpace *> &fes,
        const mfem::Array<int> &offsets_,
        mfem::real_t &dt_,
        mfem::real_t &theta_,
        mfem::real_t alpha1_,
        bool dj_use_ads_,
        Logger &logger_)
        : Solver(offsets_.Last()),
            nBlocks(offsets_.Size()-1),
            offsets(0),
            dt(dt_),
            theta(theta_),
            alpha1(alpha1_),
            dj_use_ads(dj_use_ads_),
            logger(logger_)
    {
        fes.Copy(spaces);
        offsets.MakeRef(offsets_);
    }

    void SetVerbose(bool v) { verbose = v; }

    void SetPressurePreconditioner(mfem::HypreParMatrix *MpMat, mfem::HypreParMatrix *SpMat)
    {
        delete MpSolver; delete MpPrec; delete SpOrthoSolver; delete SpSolver;

        MpSolver = new mfem::CGSolver(MPI_COMM_WORLD);
        MpSolver->SetOperator(*MpMat);
        MpSolver->SetRelTol(1e-8);
        MpSolver->SetAbsTol(1e-12);
        MpSolver->SetMaxIter(10);
        MpSolver->SetPrintLevel(mfem::IterativeSolver::PrintLevel().None());

        MpPrec = new mfem::HypreSmoother(*MpMat);
        MpPrec->SetType(mfem::HypreSmoother::GS, 6); // Symmetric Gauss-Seidel
        MpSolver->SetPreconditioner(*MpPrec);

        // Pure-Neumann pressure Laplacian: 2 AMG V-cycles, wrapped in an
        // OrthoSolver to remove the constant null space.
        SpSolver = new mfem::HypreBoomerAMG(*SpMat);
        SpSolver->SetMaxIter(2);
        SpSolver->SetCycleType(1);
        SpSolver->SetRelaxType(6); // Symmetric Gauss-Seidel
        SpSolver->SetMaxLevels(25);
        SpSolver->SetStrengthThresh(0.7);  // Value of 0.7 automatically assigned by MOOSE for 3D problems
        SpSolver->SetPrintLevel(0);

        SpOrthoSolver = new mfem::OrthoSolver(spaces[3]->GetComm());
        SpOrthoSolver->SetSolver(*SpSolver);
    }

    void SetElectricPotentialPreconditioner(mfem::HypreParMatrix *MphiMat)
    {
        delete MphiSolver; delete MphiPrec;

        MphiSolver = new mfem::CGSolver(MPI_COMM_WORLD);
        MphiSolver->SetOperator(*MphiMat);
        MphiSolver->SetRelTol(1e-8);
        MphiSolver->SetMaxIter(10);
        MphiSolver->SetPrintLevel(mfem::IterativeSolver::PrintLevel().None());

        MphiPrec = new mfem::HypreSmoother(*MphiMat);
        MphiPrec->SetType(mfem::HypreSmoother::GS, 6);
        MphiSolver->SetPreconditioner(*MphiPrec);
    }

    void SetVelocityPreconditioner(mfem::HypreParMatrix *FkMat, mfem::HypreParMatrix *BtMat)
    {
        delete FkSolver; delete FkPrec;

        FkSolver = new mfem::GMRESSolver(MPI_COMM_WORLD);
        FkSolver->SetRelTol(1e-3);
        FkSolver->SetMaxIter(200);
        FkSolver->SetKDim(50);
        FkSolver->SetPrintLevel(mfem::IterativeSolver::PrintLevel().Errors());

        // Algorithm 4.1 uses additive Schwarz; BoomerAMG is used here.  The
        // velocity is a vector H1 field ordered byNODES, so use systems AMG.
        FkPrec = new mfem::HypreBoomerAMG(*FkMat);
        FkPrec->SetSystemsOptions(spaces[2]->GetVDim(), spaces[2]->GetOrdering() == mfem::Ordering::byNODES);
        FkPrec->SetPrintLevel(0);
        FkPrec->SetCycleType(2);
        FkPrec->SetRelaxType(6);
        FkPrec->SetMaxLevels(25);
        FkSolver->SetPreconditioner(*FkPrec);
        FkSolver->SetOperator(*FkMat);

        Bt = BtMat;
    }

    void SetCurrentDensityPreconditioner(mfem::HypreParMatrix *DjMat, mfem::HypreParMatrix *GtMat, mfem::HypreParMatrix *KtMat)
    {
        delete DjSolver; delete DjPrec;
        DjSolver = nullptr; DjPrec = nullptr;

        // Option set in parameters.hpp.
        if (dj_use_ads)
        {
            // Algorithm 4.1: 5 CG iterations with the HX (ADS) preconditioner.
            auto *cg = new mfem::CGSolver(MPI_COMM_WORLD);
            cg->SetOperator(*DjMat);
            cg->SetRelTol(1e-12);
            cg->SetMaxIter(5);
            cg->SetPrintLevel(mfem::IterativeSolver::PrintLevel().None());
            DjPrec = new mfem::HypreADS(*DjMat, spaces[0]);
            DjPrec->SetPrintLevel(0);
            cg->SetPreconditioner(*DjPrec);
            DjSolver = cg;
        }
        else
        {
#ifdef MFEM_USE_MUMPS
            auto *mumps = new mfem::MUMPSSolver(MPI_COMM_WORLD);
            mumps->SetPrintLevel(0);
            mumps->SetOperator(*DjMat);
            DjSolver = mumps;
#else
            MFEM_ABORT("MFEM was built without MUMPS: use dj_solver = ads.");
#endif
        }

        Gt = GtMat;
        Kt = KtMat;
    }

    int GetVelocityIterations() const { return FkSolver ? FkSolver->GetNumIterations() : 0; }

    // Apply the preconditioner as definied in algorithm 4.1 of Li et al. 2019.
    virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const override
    {
        // Create block views of the input and output vectors
        mfem::BlockVector xblock(const_cast<mfem::Vector&>(x), offsets);
        mfem::BlockVector yblock(y, offsets);

        // Pressure solve.
        mfem::Vector &yp = yblock.GetBlock(3);
        mfem::Vector rp = xblock.GetBlock(3);
        mfem::Vector xi(rp.Size()), eta(rp.Size());
        xi = 0.0, eta = 0.0;

        MpSolver->Mult(rp, xi);  // xi = Mp^-1 (rp)
        SpOrthoSolver->Mult(rp, eta);  // eta = Sp^-1 (rp)

        // Note: this multiplication of eta by 2/tau is NOT described in algorithm 4.1,
        // but it is required.
        mfem::Vector eta2tau(rp.Size());
        mfem::real_t spCoeff = 1.0 / (theta * dt);
        eta2tau = eta;
        eta2tau *= spCoeff;
        yp = xi;
        yp *= alpha1;
        yp += eta2tau;
        yp *= -1.0;

        if (verbose) { logger << "||xi|| = " << GNorm(xi) << std::endl; }
        if (verbose) { logger << "||eta|| = " << GNorm(eta) << std::endl; }
        if (verbose) { logger << "||yp|| = " << GNorm(yp) << std::endl; }

        // Electric potential solve.
        mfem::Vector &yphi = yblock.GetBlock(1);
        mfem::Vector rphi = xblock.GetBlock(1);
        MphiSolver->Mult(rphi, yphi);  // y_phi = Mphi^-1 (-r_phi)
        yphi *= -1.0;

        if (verbose) { logger << "||yphi|| = " << GNorm(yphi) << std::endl; }

        // Velocity solve.
        mfem::Vector ru = xblock.GetBlock(2);

        mfem::Vector BtYp(ru.Size());
        Bt->Mult(yp, BtYp);
        ru -= BtYp; // Get right hand side of Fk yu = ru - Bt * yp

        if (verbose) { logger << "||BtYp|| = " << GNorm(BtYp) << std::endl; }
        if (verbose) { logger << "||ru|| = " << GNorm(ru) << std::endl; }

        mfem::Vector &yu = yblock.GetBlock(2);
        yu = 0.0;
        FkSolver->Mult(ru, yu); // Solve yu = Fk^-1 (ru - Bt * yp)

        if (verbose) { logger << "||yu|| = " << GNorm(yu) << std::endl; }

        // Current density solve.
        mfem::Vector rj = xblock.GetBlock(0);
        mfem::Vector GtYphi(rj.Size()), KtYu(rj.Size());

        Gt->Mult(yphi, GtYphi);
        GtYphi *= 2.0;
        if (verbose) { logger << "||GtYphi|| = " << GNorm(GtYphi) << std::endl; }

        Kt->Mult(yu, KtYu);
        KtYu *= 2.0;
        if (verbose) { logger << "||KtYu|| = " << GNorm(KtYu) << std::endl; }

        rj -= GtYphi;
        rj -= KtYu;
        if (verbose) { logger << "||rj|| = " << GNorm(rj) << std::endl; }

        mfem::Vector &yj = yblock.GetBlock(0);
        yj = 0.0;
        DjSolver->Mult(rj, yj); // yj = Dj^-1 (rj - 2 Gt * y_phi - 2 Kt * y_u)

        if (verbose) { logger << "||yj|| = " << GNorm(yj) << std::endl; }

        if (verbose) { logger << "||rphi|| = " << GNorm(rphi) << std::endl; }
        if (verbose) { logger << "||rp|| = " << GNorm(rp) << std::endl; }

        if (verbose) { logger << "||y|| = " << GNorm(y) << std::endl; }
    }

    virtual void SetOperator(const mfem::Operator &op) override { }

    virtual ~LiPreconditioner()
    {
        delete MpSolver;
        delete MpPrec;
        delete SpOrthoSolver;
        delete SpSolver;
        delete MphiSolver;
        delete MphiPrec;
        delete FkSolver;
        delete FkPrec;
        delete DjSolver;
        delete DjPrec;
    }
};