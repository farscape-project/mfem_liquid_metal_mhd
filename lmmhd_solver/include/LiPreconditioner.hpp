#include "mfem.hpp"
#include "constants.hpp"

using namespace mfem;
using namespace std;

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
//
//  4. Solve Dj y_j = r_j - 2 G^T y_phi - 2 K^T y_u by 5 iterations of CG solver with 
//     the HX preconditioner.
//


class OperatorSolver : public Solver
{
    Operator *A;
public:
    OperatorSolver(Operator *A_) : Solver(A_->Height(), A_->Width()), A(A_) {}
    
    virtual void Mult(const Vector &x, Vector &y) const override { A->Mult(x, y); }

    virtual void SetOperator(const Operator &op) override { }
};

class LiPreconditioner : public Solver
{
protected:
    Array<ParFiniteElementSpace *> spaces;
    int nBlocks;
    Array<int> offsets;
    std::vector<std::vector<Solver*>> solvers; 
    bool owns_blocks;
    real_t dt;
    Array<int> USE_MUMPS;

    // Pressure preconditioner solvers.
    //CGSolver *MpSolver;
    MUMPSSolver *MpSolver;
    HypreSmoother *MpPrec;

    //HypreBoomerAMG *SpSolver;
    MUMPSSolver *SpSolver;
    OrthoSolver *SpOrthoSolver;

    OperatorSolver *Lp;
    HypreBoomerAMG *LpSolver;

    // Electric potential preconditioner solvers.
    //CGSolver *MphiSolver;
    MUMPSSolver *MphiSolver;
    HypreSmoother *MphiPrec;

    // Velocity preconditioner solvers.
    //GMRESSolver *FkSolver;
    MUMPSSolver *FkSolver;
    //HypreADS *FkPrec;
    //HypreBoomerAMG *FkPrec;


    HypreParMatrix *Bt = nullptr;

    // Current density preconditioner solvers.
    //CGSolver *DjSolver;
    MUMPSSolver *DjSolver;
    HypreADS *DjPrec;

    HypreParMatrix *Gt = nullptr;
    HypreParMatrix *Kt = nullptr;

    real_t scaleMp;
    real_t scaleSp;

    // Required for MeanZero function.
    mutable ParLinearForm *mass_lf = nullptr;
    mutable ConstantCoefficient onecoeff;
    mutable Vector one_vec, mass_vec;
    mutable real_t volume = 0.0;

    inline void CalculateAndSubtractMean(Vector &p) const
    {

        if (mass_lf == nullptr)
        {
            onecoeff.constant = 1.0;

            // Set up integral.
            mass_lf = new ParLinearForm(spaces[3]);
            auto *dlfi = new DomainLFIntegrator(onecoeff);
            mass_lf->AddDomainIntegrator(dlfi);
            mass_lf->Assemble();
            mass_lf->ParallelAssemble(mass_vec);

            // Do volume integral.
            ParGridFunction one_gf(spaces[3]);
            one_gf.ProjectCoefficient(onecoeff);
            one_gf.GetTrueDofs(one_vec);
            volume = InnerProduct(mass_vec, one_vec);
        }

        // Integral of p.
        real_t integral = InnerProduct(mass_vec, p);
        // Divide by volume to get mean.
        real_t mean = integral / volume;
        // Subtract mean from solution.
        p.Add(-mean, one_vec);
    }


public:
    // Constructor
    LiPreconditioner(Array<ParFiniteElementSpace *> &fes, const Array<int> &offsets_, real_t &dt_, Array<int> USE_MUMPS_, bool owns_blocks_ = false)
        : Solver(offsets_.Last()), nBlocks(offsets_.Size()-1),
          offsets(0), owns_blocks(owns_blocks_), dt(dt_), USE_MUMPS(USE_MUMPS_)
    {
        fes.Copy(spaces);
        offsets.MakeRef(offsets_);
    }


    void SetPressurePreconditioner(HypreParMatrix *MpMat, HypreParMatrix *SpMat)
    {
        /*MpSolver = new CGSolver(MPI_COMM_WORLD);
        MpSolver->SetOperator(*MpMat);

        MpSolver->SetRelTol(1e-8);
        MpSolver->SetAbsTol(1e-12);
        MpSolver->SetMaxIter(10);
        MpSolver->SetPrintLevel(-1); // Suppress output.

        MpPrec = new HypreSmoother(*MpMat);
        MpPrec->SetType(HypreSmoother::GS, 6); // Symmetric Gauss-Seidel
        MpSolver->SetPreconditioner(*MpPrec);*/

        MpSolver = new MUMPSSolver(MPI_COMM_WORLD);
        MpSolver->SetOperator(*MpMat);

        //SpSolver = new HypreBoomerAMG(*SpMat);
        SpSolver = new MUMPSSolver(MPI_COMM_WORLD);
        SpSolver->SetPrintLevel(-1);
        //SpSolver->SetMaxIter(2);

        SpSolver->SetOperator(*SpMat);

        
        //SpSolver->SetCycleType(2);
        //SpSolver->SetRelaxType(6); // Symmetric Gauss-Seidel
        //SpSolver->SetMaxLevels(25);
        //SpSolver->SetStrengthThresh(0.7);  // Value of 0.7 automatically assigned by MOOSE for 3D problems
        //SpSolver->SetElasticityOptions(spaces[3]);

        //SpSolver = new HypreBoomerAMG(*SpMat);


        // Attempting using OrthoSolver just for Sp, but perhaps not sufficient.  It may 
        // be required to wrap around whole preconditioner.
        //SpOrthoSolver = new OrthoSolver(spaces[3]->GetComm());
        //SpOrthoSolver->SetSolver(*SpSolver);


        // Scaling the pressure preconditioner values by largest value in Mp and Sp, 
        // respectively.
        /*real_t MpNorm = 0.0;
        HypreParVector diag;
        diag = 0.0;
        MpMat->GetDiag(diag);

        for (int i = 0; i < diag.Size(); i++)
        {
            if (diag(i) > MpNorm) MpNorm = diag(i);
        }

        real_t SpNorm = 0.0;
        diag = 0.0;
        SpMat->GetDiag(diag);

        for (int i = 0; i < diag.Size(); i++)
        {
            if (diag(i) > SpNorm) SpNorm = diag(i);
        }

        scaleMp = 1.0 / MpNorm;
        scaleSp = 1.0 / SpNorm;*/


        /*HypreParMatrix *Lp = Add(1.0 * scaleMp, *MpMat, (2.0/dt) * scaleSp, *SpMat);

        LpSolver = new HypreBoomerAMG(*Lp);
        LpSolver->SetMaxIter(2);
        LpSolver->SetMaxLevels(25);
        LpSolver->SetCycleType(2);
        LpSolver->SetRelaxType(6);
        LpSolver->SetMaxIter(5);
        LpSolver->SetStrengthThresh(0.7);*/


    }

    void SetElectricPotentialPreconditioner(HypreParMatrix *MphiMat)
    {
        /*MphiSolver = new CGSolver(MPI_COMM_WORLD);
        MphiSolver->SetOperator(*MphiMat);

        MphiSolver->SetRelTol(1e-8);
        MphiSolver->SetMaxIter(10);
        MphiSolver->SetPrintLevel(-1); // Suppress output.

        MphiPrec = new HypreSmoother(*MphiMat);
        MphiPrec->SetType(HypreSmoother::GS, 6);
        MphiSolver->SetPreconditioner(*MphiPrec);*/

        MphiSolver = new MUMPSSolver(MPI_COMM_WORLD);
        MphiSolver->SetOperator(*MphiMat);
    }

    void SetVelocityPreconditioner(HypreParMatrix *FkMat, HypreParMatrix *BtMat)
    {
        //FkSolver = new GMRESSolver(MPI_COMM_WORLD);
        FkSolver = new MUMPSSolver(MPI_COMM_WORLD);
        FkSolver->SetOperator(*FkMat);
        //FkSolver->SetRelTol(1e-3); // Temporarily increasing values for testing.
        //FkSolver->SetMaxIter(500);  // Temporarily increasing values for testing.
        //FkSolver->SetPrintLevel(-1);
        
        // Additive Schwarz?!
        //FkPrec = new HypreADS(*FkMat, spaces[2]);
        //FkPrec = new HypreBoomerAMG(*FkMat);
        //FkPrec->SetPrintLevel(-1);
        //FkPrec->SetCycleType(2);
        //FkPrec->SetRelaxType(6);
        //FkPrec->SetMaxLevels(25);
        //FkSolver->SetPreconditioner(*FkPrec);

        //FkSolver = new MUMPSSolver(MPI_COMM_WORLD);
        //FkSolver->SetOperator(*FkMat);

        Bt = BtMat;
    }

    void UpdateVelocityPreconditioner(HypreParMatrix *FkMat)
    {
        FkSolver->SetOperator(*FkMat);
    }


    void SetCurrentDensityPreconditioner(HypreParMatrix *DjMat, HypreParMatrix *GtMat, HypreParMatrix *KtMat)
    {
        /*DjSolver = new CGSolver(MPI_COMM_WORLD);
        DjSolver->SetOperator(*DjMat);
        DjSolver->SetRelTol(1e-8);
        DjSolver->SetMaxIter(5);
        DjSolver->SetPrintLevel(-1); // Suppress output.

        DjPrec = new HypreADS(*DjMat, spaces[0]);
        DjSolver->SetPreconditioner(*DjPrec);*/

        DjSolver = new MUMPSSolver(MPI_COMM_WORLD);
        DjSolver->SetOperator(*DjMat);

        Gt = GtMat;
        Kt = KtMat;
    }

    // Apply the preconditioner as definied in algorithm 4.1 of Li et al. 2019.
    virtual void Mult(const Vector &x, Vector &y) const override
    {
        // Create block views of the input and output vectors
        BlockVector xblock(const_cast<Vector&>(x), offsets);
        BlockVector yblock(y, offsets);

        // Pressure solve.
        Vector &yp = yblock.GetBlock(3);
        Vector rp = xblock.GetBlock(3);
        Vector xi(rp.Size()), eta(rp.Size());
        xi = 0.0, eta = 0.0;

        MpSolver->Mult(rp, xi);  // xi = Mp^-1 (rp)
        SpSolver->Mult(rp, eta);  // eta = Sp^-1 (rp)


        // Scaling the pressure preconditioner values by largest value in Mp and Sp, 
        // respectively.
        //xi *= scaleMp;
        //eta *= scaleSp;

        // Note: this multiplication of eta by 2/tau is NOT described in algorithm 4.1.
        Vector eta2tau(rp.Size());
        //real_t beta = 1.0;
        real_t spCoeff = 2.0;
        // The coefficient is NOT divided by dt as 
        // this is taken into account in ImplicitSolve.
        eta2tau = eta;
        eta2tau *= spCoeff;
        yp = xi;  yp *= alpha1;  yp += eta2tau;  yp *= -1.0;

        CalculateAndSubtractMean(yp);

        std::cout << "||xi|| = " << xi.Norml2() << std::endl;
        std::cout << "||eta|| = " << eta.Norml2() << std::endl;
        std::cout << "||yp|| = " << yp.Norml2() << std::endl;
        
        // Testing effect of yp on solution.
        //yp *= 0.0;

        // Testing solving Mp and Sp together, rather than separately.
        //LpSolver->Mult(rp, yp);
        //yp *= -1.0;

        // Electric potential solve.
        Vector &yphi = yblock.GetBlock(1);
        Vector rphi = xblock.GetBlock(1);
        MphiSolver->Mult(rphi, yphi);  // y_phi = Mphi^-1 (-r_phi)
        yphi *= -1.0;
        //yphi *= 5.0;

        CalculateAndSubtractMean(yphi);

        std::cout << "||yphi|| = " << yphi.Norml2() << std::endl;

        // Velocity solve.
        Vector ru = xblock.GetBlock(2);

        Vector BtYp(ru.Size());
        Bt->Mult(yp, BtYp);
        ru -= BtYp; // Get right hand side of Fk yu = ru - Bt * yp

        std::cout << "||BtYp|| = " << BtYp.Norml2() << std::endl;
        std::cout << "||ru|| = " << ru.Norml2() << std::endl;

        Vector &yu = yblock.GetBlock(2);
        yu = 0.0;
        FkSolver->Mult(ru, yu); // Solve yu = Fk^-1 (ru - Bt * yp)

        std::cout << "||yu|| = " << yu.Norml2() << std::endl;

        // Current density solve.
        Vector rj = xblock.GetBlock(0);
        Vector GtYphi(rj.Size()), KtYu(rj.Size());

        Gt->Mult(yphi, GtYphi);
        GtYphi *= 2.0;
        std::cout << "||GtYphi|| = " << GtYphi.Norml2() << std::endl;

        Kt->Mult(yu, KtYu);
        KtYu *= 2.0;
        std::cout << "||KtYu|| = " << KtYu.Norml2() << std::endl;

        rj -= GtYphi;
        rj -= KtYu;
        std::cout << "||rj|| = " << rj.Norml2() << std::endl;


        Vector &yj = yblock.GetBlock(0);
        yj = 0.0;
        DjSolver->Mult(rj, yj); // yj = Dj^-1 (rj - 2 Gt * y_phi - 2 Kt * y_u)
        //yj *= 10.0;
        std::cout << "||yj|| = " << yj.Norml2() << std::endl;

        std::cout << "||rphi|| = " << rphi.Norml2() << std::endl;
        std::cout << "||rp|| = " << rp.Norml2() << std::endl;

        std::cout << "||y|| = " << y.Norml2() << std::endl;

    }

    virtual void SetOperator(const Operator &op) override { }

    virtual ~LiPreconditioner()
    {
        delete MpSolver;
        delete SpSolver;
        delete MphiSolver;
        delete FkSolver;
        delete DjSolver;
    }
};