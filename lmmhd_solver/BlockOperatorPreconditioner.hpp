#include "mfem.hpp"
using namespace mfem;
using namespace std;

class BlockOperatorPreconditioner : public Solver
{
protected:
    int nBlocks;
    Array<int> offsets;
    std::vector<std::vector<Operator*>> op; 
    bool owns_blocks;

public:
    // Constructor
    BlockOperatorPreconditioner(const Array<int> &offsets_, bool owns_blocks_ = false)
        : Solver(offsets_.Last()), nBlocks(offsets_.Size()-1),
          offsets(0), owns_blocks(owns_blocks_)
    {
        offsets.MakeRef(offsets_);
        op.resize(nBlocks);
        for (int i=0; i<nBlocks; ++i) op[i].resize(nBlocks, nullptr);
    }

    // Set a block operator (can be off-diagonal)
    void SetBlock(int i, int j, Operator *opt)
    {
        // Optional: verify block dimensions match
        int hi = offsets[i+1]-offsets[i];
        int hj = offsets[j+1]-offsets[j];
        MFEM_VERIFY(opt->Height() == hi && opt->Width() == hj,
                    "Operator dimensions do not match block size");

        op[i][j] = opt;
    }

    // Apply the block operator as a preconditioner
    virtual void Mult(const Vector &x, Vector &y) const override
    {
        MFEM_ASSERT(x.Size() == width, "incorrect input Vector size");
        MFEM_ASSERT(y.Size() == height, "incorrect output Vector size");

        BlockVector xblock(const_cast<Vector&>(x), offsets);
        BlockVector yblock(y, offsets);

        for (int i=0; i<nBlocks; ++i)
        {
            yblock.GetBlock(i) = 0.0;
            bool diagonal_exists = false;

            for (int j=0; j<nBlocks; ++j)
            {
                if (op[i][j])
                {
                    Vector tmp(op[i][j]->Height());
                    op[i][j]->Mult(xblock.GetBlock(j), tmp);
                    yblock.GetBlock(i) += tmp;
                    if (i == j) diagonal_exists = true;
                }
            }

            if (!diagonal_exists)
            {
                // Set to identity if diagonal block is not set.
                yblock.GetBlock(i) = xblock.GetBlock(i);
            }
        }
    }

    // Apply transpose
    virtual void MultTranspose(const Vector &x, Vector &y) const override
    {
        MFEM_ASSERT(x.Size() == height, "incorrect input Vector size");
        MFEM_ASSERT(y.Size() == width, "incorrect output Vector size");

        BlockVector xblock(const_cast<Vector&>(x), offsets);
        BlockVector yblock(y, offsets);

        for (int i=0; i<nBlocks; ++i)
        {
            yblock.GetBlock(i) = 0.0;
            bool diagonal_exists = false;

            for (int j=0; j<nBlocks; ++j)
            {
                if (op[i][j])
                {
                    Vector tmp(op[i][j]->Width());
                    op[i][j]->MultTranspose(xblock.GetBlock(j), tmp);
                    yblock.GetBlock(i) += tmp;
                }
            }

            if (!diagonal_exists)
            {
                // Set to identity if diagonal block is not set.
                yblock.GetBlock(i) = xblock.GetBlock(i);
            }
        }
    }

    virtual void SetOperator(const Operator &op) override { }

    virtual ~BlockOperatorPreconditioner()
    {
        if (owns_blocks)
        {
            for (int i=0; i<nBlocks; ++i)
                for (int j=0; j<nBlocks; ++j)
                    delete op[i][j];
        }
    }
};