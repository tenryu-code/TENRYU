#pragma once

#include <cstdint>

#if TENRYU_ENABLE_MPI
#include <mpi.h>
#endif

namespace tenryu::parallel {

class Reduction {
 public:
  explicit Reduction(int n_ranks = 1);

  double allreduce_sum(double local_value) const;
  double allreduce_min(double local_value) const;
  double allreduce_max(double local_value) const;
  std::uint64_t allreduce_min(std::uint64_t local_value) const;

  void allreduce_sum(double* data, int count) const;
  void allreduce_min(double* data, int count) const;
  void allreduce_max(double* data, int count) const;

  int64_t exscan_sum(int64_t local_value) const;

  void broadcast(double* data, int count, int root) const;

  void allgatherv(const double* sendbuf, int sendcount,
                  double* recvbuf, const int* recvcounts,
                  const int* displs) const;

  double sum(double value) const { return allreduce_sum(value); }

 private:
  int n_ranks_ = 1;
#if TENRYU_ENABLE_MPI
  MPI_Comm comm_ = MPI_COMM_WORLD;
#endif
};

}  // namespace tenryu::parallel
