// scaling_timer.hpp
// Simple named wall-clock timers for MPI scaling.
//
// Every rank must call start()/stop() for the same sections in the same
// order (true as long as all ranks run the same code path), and every rank
// must call report() at the end, because it does collective reductions.

#pragma once

#include <mpi.h>

#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>

class ScalingTimer
{
public:
   explicit ScalingTimer(MPI_Comm comm = MPI_COMM_WORLD) : comm_(comm)
   {
      MPI_Comm_rank(comm_, &rank_);
      MPI_Comm_size(comm_, &nranks_);
   }

   // sync = true puts a barrier before the timestamp so every rank starts
   // and stops together; the measured time is then true wall-clock time for
   // the section, including any load imbalance inside it.
   void start(const std::string &name, bool sync = true)
   {
      if (sync) { MPI_Barrier(comm_); }
      if (acc_.find(name) == acc_.end())
      {
         order_.push_back(name);
         acc_[name] = 0.0;
         calls_[name] = 0;
      }
      t0_[name] = MPI_Wtime();
   }

   void stop(const std::string &name, bool sync = true)
   {
      if (sync) { MPI_Barrier(comm_); }
      acc_[name] += MPI_Wtime() - t0_[name];
      calls_[name] += 1;
   }

   // Prints a table on rank 0 and appends one CSV row per section.
   // ndofs: global problem size (e.g. GlobalTrueVSize()), recorded so runs
   // at different core counts / mesh sizes can be compared.
   void report(long long ndofs, const std::string &csv_file = "timings.csv")
   {
      const int n = static_cast<int>(order_.size());
      std::vector<double> local(n), tmax(n), tmin(n), tsum(n);
      for (int i = 0; i < n; i++) { local[i] = acc_[order_[i]]; }

      MPI_Reduce(local.data(), tmax.data(), n, MPI_DOUBLE, MPI_MAX, 0, comm_);
      MPI_Reduce(local.data(), tmin.data(), n, MPI_DOUBLE, MPI_MIN, 0, comm_);
      MPI_Reduce(local.data(), tsum.data(), n, MPI_DOUBLE, MPI_SUM, 0, comm_);

      if (rank_ != 0) { return; }

      std::printf("\n=== Timings: %d ranks, %lld DOFs ===\n", nranks_, ndofs);
      std::printf("%-20s %8s %12s %12s %12s\n",
                  "section", "calls", "max [s]", "avg [s]", "min [s]");
      for (int i = 0; i < n; i++)
      {
         std::printf("%-20s %8d %12.4f %12.4f %12.4f\n",
                     order_[i].c_str(), calls_[order_[i]],
                     tmax[i], tsum[i] / nranks_, tmin[i]);
      }

      std::ifstream exists(csv_file);
      const bool write_header = !exists.good();
      exists.close();

      std::ofstream out(csv_file, std::ios::app);
      if (write_header)
      {
         out << "nranks,ndofs,section,calls,max_s,avg_s,min_s\n";
      }
      for (int i = 0; i < n; i++)
      {
         out << nranks_ << ',' << ndofs << ',' << order_[i] << ','
             << calls_[order_[i]] << ',' << tmax[i] << ','
             << tsum[i] / nranks_ << ',' << tmin[i] << '\n';
      }
   }

private:
   MPI_Comm comm_;
   int rank_ = 0, nranks_ = 1;
   std::vector<std::string> order_;
   std::map<std::string, double> t0_, acc_;
   std::map<std::string, int> calls_;
};