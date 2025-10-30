#include "mfem.hpp"
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
//  1. Compute y_p = -Lp r_p = - alpha_1 xi - eta,
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

    // Pressure preconditioner solvers.
    OperatorSolver *Mp;;
    CGSolver *MpSolver;
    HypreSmoother *MpPrec;

    OperatorSolver *Sp;;
    HypreBoomerAMG *SpSolver;

    // Electric potential preconditioner solvers.
    OperatorSolver *Mphi;
    CGSolver *MphiSolver;
    HypreSmoother *MphiPrec;

    // Velocity preconditioner solvers.
    OperatorSolver *Fk;
    GMRESSolver *FkSolver;

    HypreParMatrix *Bt = nullptr;

    // Current density preconditioner solvers.
    OperatorSolver *Dj;
    CGSolver *DjSolver;
    HypreADS *DjPrec;

    HypreParMatrix *Gt = nullptr;
    HypreParMatrix *Kt = nullptr;

public:
    // Constructor
    LiPreconditioner(Array<ParFiniteElementSpace *> &fes, const Array<int> &offsets_, bool owns_blocks_ = false)
        : Solver(offsets_.Last()), nBlocks(offsets_.Size()-1),
          offsets(0), owns_blocks(owns_blocks_)
    {
        fes.Copy(spaces);
        offsets.MakeRef(offsets_);
    }


    void SetPressurePreconditioner(HypreParMatrix *MpMat, HypreParMatrix *SpMat)
    {
        MpSolver = new CGSolver(MPI_COMM_WORLD);
        MpSolver->SetRelTol(1e-8);
        MpSolver->SetMaxIter(10);
        MpSolver->SetPrintLevel(0);

        MpSolver->SetOperator(*MpMat);

        MpPrec = new HypreSmoother(*MpMat);
        MpPrec->SetType(HypreSmoother::Jacobi);
        MpSolver->SetPreconditioner(*MpPrec);

        Mp = new OperatorSolver(MpSolver);

        SpSolver = new HypreBoomerAMG(*SpMat);
        SpSolver->SetPrintLevel(0);      
        SpSolver->SetCycleType(1);
        SpSolver->SetRelaxType(6); // 6 = Symmetric Gauss-Seidel (common choice)
        SpSolver->SetMaxLevels(25);

        SpSolver->SetOperator(*SpMat);

        Sp = new OperatorSolver(SpSolver);

    }

    void SetElectricPotentialPreconditioner(HypreParMatrix *MphiMat)
    {
        MphiSolver = new CGSolver(MPI_COMM_WORLD);
        MphiSolver->SetRelTol(1e-8);
        MphiSolver->SetMaxIter(10);
        MphiSolver->SetPrintLevel(0);

        MphiSolver->SetOperator(*MphiMat);

        MphiPrec = new HypreSmoother(*MphiMat);
        MphiPrec->SetType(HypreSmoother::Jacobi);
        MphiSolver->SetPreconditioner(*MphiPrec);

        Mphi = new OperatorSolver(MphiSolver);
    }

    void SetVelocityPreconditioner(HypreParMatrix *FkMat, HypreParMatrix *BtMat)
    {
        FkSolver = new GMRESSolver(MPI_COMM_WORLD);
        FkSolver->SetRelTol(1e-3);
        FkSolver->SetMaxIter(200);
        FkSolver->SetPrintLevel(0);
        
        FkSolver->SetOperator(*FkMat);

        Fk = new OperatorSolver(FkSolver);

        Bt = BtMat;
    }

    void SetCurrentDensityPreconditioner(HypreParMatrix *DjMat, HypreParMatrix *GTMat, HypreParMatrix *KtMat)
    {
        DjSolver = new CGSolver(MPI_COMM_WORLD);
        DjSolver->SetRelTol(1e-8);
        DjSolver->SetMaxIter(5);
        DjSolver->SetPrintLevel(0);

        DjSolver->SetOperator(*DjMat);

        DjPrec = new HypreADS(*DjMat, spaces[0]);
        DjSolver->SetPreconditioner(*DjPrec);

        Dj = new OperatorSolver(DjSolver);

        Gt = GTMat;
        Kt = KtMat;
    }

    // Apply the preconditioner as definied in algorithm 4.1 of Li et al. 2019.
    virtual void Mult(const Vector &x, Vector &y) const override
    {
        // Create block views of the input and output vectors
        BlockVector xblock(const_cast<Vector&>(x), offsets);
        BlockVector yblock(y, offsets);


        // Pressure solve.
        Vector rp = xblock.GetBlock(3);
        Vector xi(rp.Size()), eta(rp.Size());
        xi = 0.0, eta = 0.0;

        Mp->Mult(rp, xi);  // xi = Mp^-1 (rp)
        Sp->Mult(rp, eta);  // eta = Sp^-1 (rp)

        Vector &yp = yblock.GetBlock(3);
        yp = xi;
        yp *= 1.0; // alpha currently 1.0.  Update later.
        yp += eta;
        yp *= -1.0;        


        // Electric potential solve.
        Vector rphi = xblock.GetBlock(1);
        rphi *= -1.0;

        Vector &yphi = yblock.GetBlock(1);
        Mphi->Mult(rphi, yphi);  // y_phi = Mphi^-1 (-r_phi)


        // Velocity solve.
        Vector ru = xblock.GetBlock(2);

        Vector BtYp(ru.Size());
        Bt->Mult(yp, BtYp);
        ru -= BtYp; // Get right hand side of Fk yp = ru - Bt * y_p

        Vector &yu = yblock.GetBlock(2);
        yu = 0.0;
        FkSolver->Mult(ru, yu); // Solve yp = Fk^-1 (ru - Bt * y_p)


        // Current density solve.
        Vector rj = xblock.GetBlock(0);
        Vector GtYphi(rj.Size()), KtYu(rj.Size());

        Gt->Mult(yphi, GtYphi);
        GtYphi *= 2.0;
        Kt->Mult(yu, KtYu);
        KtYu *= 2.0;

        rj -= GtYphi;
        rj -= KtYu;

        Vector &yj = yblock.GetBlock(0);
        yj = 0.0;
        Dj->Mult(rj, yj); // yj = Dj^-1 (rj - Gt * y_phi - Kt * y_u)

    }

    virtual void SetOperator(const Operator &op) override { }

    virtual ~LiPreconditioner()
    {
        if (owns_blocks)
        {
            delete Mphi;
            delete Mp;
            delete Sp;
            delete Fk;
            delete Dj;
        }
    }
};