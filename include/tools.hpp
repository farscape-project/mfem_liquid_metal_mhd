#pragma once
#include "mfem.hpp"
#include <iostream>
#include <fstream>

// Print all parameter read in from input file.
void PrintParams(int nx, int ny, int nz, 
		mfem::real_t Lx, mfem::real_t Ly, mfem::real_t Lz,
        mfem::real_t clusterX, mfem::real_t clusterY, mfem::real_t clusterZ,
        mfem::real_t t_final, mfem::real_t dt, int vis_steps, 
		mfem::real_t Re, mfem::real_t N, mfem::real_t Ha);

// Print finite element space sizes.
void PrintFESpaces(int j_space_size, int phi_space_size, 
        int v_space_size, int p_space_size);

// Clustering function (symmetric around center).
mfem::real_t cluster_symmetric(mfem::real_t xi, mfem::real_t factor);

// Checkpoint function for debugging.
void checkpoint(int num);

// Relative change ||gf - gfN_1|| / ||gf|| between time levels, using the
// global Euclidean norm of the true-dof vectors (collective over the
// communicator of fespace).
mfem::real_t rel_L2_norm(const mfem::ParGridFunction &gf, const mfem::ParGridFunction &gfN_1, mfem::ParFiniteElementSpace *fespace);

class RemoveMeanProjector
{
private:

   MPI_Comm comm = MPI_COMM_NULL;
   mfem::ConstantCoefficient onecoeff;
   mfem::Vector one_vec, mass_vec;
   mfem::real_t volume = 0.0;

public:

   RemoveMeanProjector() = default;

   RemoveMeanProjector(mfem::ParFiniteElementSpace &fes);

   void RemoveMean(mfem::Vector &v) const;
};

// Helper class for outputting to log file.
class Logger
{
private:
    std::ostream &file;

public:
    Logger(std::ostream &file_) : file(file_) {}

    template <typename T>
    Logger &operator<<(const T &value)
    {
        if (mfem::Mpi::Root())
        {
            file << value;
        }
        return *this;
    }

    // Support std::endl
    Logger &operator<<(std::ostream& (*manip)(std::ostream&))
    {
        if (mfem::Mpi::Root())
        {
            manip(file);
        }
        return *this;
    }
};

// Helper class for accessing FGMRES residuals and outputting to log file.
class FGMRESLogMonitor : public mfem::IterativeSolverMonitor
{
private:
    Logger logger;

public:
    FGMRESLogMonitor(Logger &logger_)
        : logger(logger_)
    {}

    virtual void MonitorResidual(int it,
                                 mfem::real_t norm,
                                 const mfem::Vector &r,
                                 bool final) override
    {
        if (mfem::Mpi::Root())
        {
            logger << "FGMRES iteration : "
                     << it
                     << " || r || = "
                     << norm
                     << std::endl;
        }
    }
};