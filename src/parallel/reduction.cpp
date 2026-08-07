#include <vector>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <atomic>
#include "parallel/reduction.hpp"

#include <algorithm>
#include <string>

#include "core/error.hpp"

namespace {
// TENRYU_COMM_PROFILE (design doc §6q.3): env-gated per-entry-point call
// counters. Zero effect without the env (one branch per call).
struct CommProfSlot {
  const char* name;
  std::atomic<long> calls{0};
  std::atomic<double> total_s{0.0};
};
inline bool comm_prof_enabled() {
  static const bool on = std::getenv("TENRYU_COMM_PROFILE") != nullptr;
  return on;
}
inline void comm_prof_add(CommProfSlot& slot, const double dt_s) {
  slot.calls.fetch_add(1, std::memory_order_relaxed);
  double cur = slot.total_s.load(std::memory_order_relaxed);
  while (!slot.total_s.compare_exchange_weak(cur, cur + dt_s)) {
  }
}
struct CommProfRegistry {
  std::vector<CommProfSlot*> slots;
  ~CommProfRegistry() {
    if (!comm_prof_enabled()) {
      return;
    }
    for (CommProfSlot* s : slots) {
      if (s->calls.load() == 0) {
        continue;
      }
      const long c = s->calls.load();
      const double t = s->total_s.load();
      std::fprintf(stderr,
                   "[comm-profile] %s calls=%ld total=%.3fs avg=%.3fms\n",
                   s->name, c, t, c > 0 ? 1e3 * t / c : 0.0);
    }
  }
};
CommProfRegistry& comm_prof_registry() {
  static CommProfRegistry reg;
  return reg;
}
struct CommProfTimer {
  CommProfSlot& slot;
  std::chrono::steady_clock::time_point t0;
  bool armed;
  explicit CommProfTimer(CommProfSlot& s) : slot(s), armed(comm_prof_enabled()) {
    if (armed) {
      t0 = std::chrono::steady_clock::now();
    }
  }
  ~CommProfTimer() {
    if (armed) {
      const auto t1 = std::chrono::steady_clock::now();
      comm_prof_add(slot,
                    std::chrono::duration<double>(t1 - t0).count());
    }
  }
};
}  // namespace


namespace tenryu::parallel {

#if TENRYU_ENABLE_MPI
namespace {

double allreduce_scalar(const MPI_Comm comm, const double local_value,
                        const MPI_Op op, const char* op_name) {
  double reduced_value = local_value;
  const int rc = MPI_Allreduce(&local_value, &reduced_value, 1, MPI_DOUBLE, op, comm);
  TENRYU_ASSERT(rc == MPI_SUCCESS, std::string("MPI_Allreduce(") + op_name + ") failed");
  return reduced_value;
}

std::uint64_t allreduce_scalar_u64(const MPI_Comm comm,
                                   const std::uint64_t local_value,
                                   const MPI_Op op,
                                   const char* op_name) {
  std::uint64_t reduced_value = local_value;
  const int rc = MPI_Allreduce(&local_value, &reduced_value, 1, MPI_UINT64_T,
                               op, comm);
  TENRYU_ASSERT(rc == MPI_SUCCESS,
                std::string("MPI_Allreduce(") + op_name + ") failed");
  return reduced_value;
}

void allreduce_vector_in_place(const MPI_Comm comm, double* data,
                               const int count, const MPI_Op op, const char* op_name) {
  if (count <= 0) {
    return;
  }
  const int rc = MPI_Allreduce(MPI_IN_PLACE, data, count, MPI_DOUBLE, op, comm);
  TENRYU_ASSERT(rc == MPI_SUCCESS, std::string("MPI_Allreduce(") + op_name + ") failed");
}

}  // namespace
#endif

Reduction::Reduction(const int n_ranks) : n_ranks_(n_ranks > 1 ? n_ranks : 1) {}

double Reduction::allreduce_sum(const double local_value) const {
  static CommProfSlot comm_prof_slot_ar_sum1{"ar_sum1"};
  static const bool comm_prof_reg_ar_sum1 =
      (comm_prof_registry().slots.push_back(&comm_prof_slot_ar_sum1),
       true);
  (void)comm_prof_reg_ar_sum1;
  CommProfTimer comm_prof_timer_ar_sum1(comm_prof_slot_ar_sum1);
  if (n_ranks_ == 1) {
    return local_value;
  }
#if TENRYU_ENABLE_MPI
  return allreduce_scalar(comm_, local_value, MPI_SUM, "SUM");
#else
  return local_value;
#endif
}

double Reduction::allreduce_min(const double local_value) const {
  static CommProfSlot comm_prof_slot_ar_min1{"ar_min1"};
  static const bool comm_prof_reg_ar_min1 =
      (comm_prof_registry().slots.push_back(&comm_prof_slot_ar_min1),
       true);
  (void)comm_prof_reg_ar_min1;
  CommProfTimer comm_prof_timer_ar_min1(comm_prof_slot_ar_min1);
  if (n_ranks_ == 1) {
    return local_value;
  }
#if TENRYU_ENABLE_MPI
  return allreduce_scalar(comm_, local_value, MPI_MIN, "MIN");
#else
  return local_value;
#endif
}

double Reduction::allreduce_max(const double local_value) const {
  static CommProfSlot comm_prof_slot_ar_max1{"ar_max1"};
  static const bool comm_prof_reg_ar_max1 =
      (comm_prof_registry().slots.push_back(&comm_prof_slot_ar_max1),
       true);
  (void)comm_prof_reg_ar_max1;
  CommProfTimer comm_prof_timer_ar_max1(comm_prof_slot_ar_max1);
  if (n_ranks_ == 1) {
    return local_value;
  }
#if TENRYU_ENABLE_MPI
  return allreduce_scalar(comm_, local_value, MPI_MAX, "MAX");
#else
  return local_value;
#endif
}

std::uint64_t Reduction::allreduce_min(const std::uint64_t local_value) const {
  if (n_ranks_ == 1) {
    return local_value;
  }
#if TENRYU_ENABLE_MPI
  return allreduce_scalar_u64(comm_, local_value, MPI_MIN, "MIN");
#else
  return local_value;
#endif
}

void Reduction::allreduce_sum(double* data, const int count) const {
  static CommProfSlot comm_prof_slot_ar_sumN{"ar_sumN"};
  static const bool comm_prof_reg_ar_sumN =
      (comm_prof_registry().slots.push_back(&comm_prof_slot_ar_sumN),
       true);
  (void)comm_prof_reg_ar_sumN;
  CommProfTimer comm_prof_timer_ar_sumN(comm_prof_slot_ar_sumN);
  if (n_ranks_ == 1 || count <= 0) {
    return;
  }
#if TENRYU_ENABLE_MPI
  allreduce_vector_in_place(comm_, data, count, MPI_SUM, "SUM");
#else
  (void)data;
#endif
}

void Reduction::allreduce_min(double* data, const int count) const {
  if (n_ranks_ == 1 || count <= 0) {
    return;
  }
#if TENRYU_ENABLE_MPI
  allreduce_vector_in_place(comm_, data, count, MPI_MIN, "MIN");
#else
  (void)data;
#endif
}

void Reduction::allreduce_max(double* data, const int count) const {
  if (n_ranks_ == 1 || count <= 0) {
    return;
  }
#if TENRYU_ENABLE_MPI
  allreduce_vector_in_place(comm_, data, count, MPI_MAX, "MAX");
#else
  (void)data;
#endif
}

int64_t Reduction::exscan_sum(const int64_t local_value) const {
  if (n_ranks_ == 1) {
    return 0;
  }
#if TENRYU_ENABLE_MPI
  int64_t rank_offset = 0;
  const int exscan_rc =
      MPI_Exscan(&local_value, &rank_offset, 1, MPI_INT64_T, MPI_SUM, comm_);
  TENRYU_ASSERT(exscan_rc == MPI_SUCCESS, "MPI_Exscan(SUM) failed");

  int rank = 0;
  const int rank_rc = MPI_Comm_rank(comm_, &rank);
  TENRYU_ASSERT(rank_rc == MPI_SUCCESS, "MPI_Comm_rank(exscan) failed");
  if (rank == 0) {
    return 0;
  }
  return rank_offset;
#else
  (void)local_value;
  return 0;
#endif
}

void Reduction::broadcast(double* data, const int count, const int root) const {
  if (n_ranks_ == 1 || count <= 0) {
    return;
  }
#if TENRYU_ENABLE_MPI
  const int rc = MPI_Bcast(data, count, MPI_DOUBLE, root, comm_);
  TENRYU_ASSERT(rc == MPI_SUCCESS, "MPI_Bcast(double) failed");
#else
  (void)data;
  (void)root;
#endif
}

void Reduction::allgatherv(const double* sendbuf, const int sendcount,
                           double* recvbuf, const int* recvcounts,
                           const int* displs) const {
  static CommProfSlot comm_prof_slot_allgatherv{"allgatherv"};
  static const bool comm_prof_reg_allgatherv =
      (comm_prof_registry().slots.push_back(&comm_prof_slot_allgatherv),
       true);
  (void)comm_prof_reg_allgatherv;
  CommProfTimer comm_prof_timer_allgatherv(comm_prof_slot_allgatherv);
  if (n_ranks_ == 1) {
    if (sendcount <= 0) {
      return;
    }
    const int offset = displs != nullptr ? displs[0] : 0;
    std::copy_n(sendbuf, sendcount, recvbuf + offset);
    return;
  }

#if TENRYU_ENABLE_MPI
  const int rc = MPI_Allgatherv(sendbuf, sendcount, MPI_DOUBLE, recvbuf, recvcounts,
                                displs, MPI_DOUBLE, comm_);
  TENRYU_ASSERT(rc == MPI_SUCCESS, "MPI_Allgatherv(double) failed");
#else
  if (sendcount <= 0) {
    return;
  }
  const int offset = displs != nullptr ? displs[0] : 0;
  std::copy_n(sendbuf, sendcount, recvbuf + offset);
  (void)recvcounts;
#endif
}

}  // namespace tenryu::parallel
