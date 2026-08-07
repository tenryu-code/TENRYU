#pragma once

#include <cmath>
#include <cfloat>

namespace tenryu::materials {

/// Device-side read-only view of an IONMIX opacity table.
/// All pointers refer to GPU device memory uploaded by the host.
/// Replicates IonmixOpacityData::interpolate_kappa() logic on GPU.
struct IonmixOpacityDeviceView {
    static constexpr double kKappaFloor = 1.0e-100;
    static constexpr double kFloorAwareThreshold = 1.0e-30;

    const double* log_temps = nullptr;     ///< [ntemp] pre-computed log(T_eV) grid
    const double* log_numdens = nullptr;   ///< [ndens] pre-computed log(n_i [cm^-3]) grid
    const double* kappa_PA = nullptr;      ///< [ngroups * ndens * ntemp] Planck absorption [cm^2/g]
    const double* kappa_PE = nullptr;      ///< [ngroups * ndens * ntemp] Planck emission  [cm^2/g]
    const double* kappa_R  = nullptr;      ///< [ngroups * ndens * ntemp] Rosseland        [cm^2/g]
    int ntemp   = 0;
    int ndens   = 0;
    int ngroups = 0;
    double log_T_min  = 0.0;   ///< log(temps_eV.front())
    double log_T_max  = 0.0;   ///< log(temps_eV.back())
    double log_ni_min = 0.0;   ///< log(numdens_cm3.front())
    double log_ni_max = 0.0;   ///< log(numdens_cm3.back())
    double T_min = 0.0;        ///< temps_eV.front()  (linear, for FD boundary)
    double T_max = 0.0;        ///< temps_eV.back()   (linear, for FD boundary)

    /// Flat index into kappa tables: layout [group][dens][temp], T-fastest.
    __device__ __forceinline__ int flat_index(int g, int d, int t) const {
        return g * ndens * ntemp + d * ntemp + t;
    }

    /// Binary search bracket: find (i0, i1) such that grid[i0] <= x < grid[i1],
    /// plus interpolation fraction frac in [0,1].
    /// Replicates host bracket_index() + fraction computation.
    __device__ __forceinline__ void bracket(
        const double* grid, int n, double x,
        int& i0, int& i1, double& frac) const
    {
        // Edge cases: clamp to boundary
        if (n < 2) { i0 = 0; i1 = 0; frac = 0.0; return; }
        if (x <= grid[0]) {
            i0 = 0; i1 = 1; frac = 0.0;
            return;
        }
        if (x >= grid[n - 1]) {
            i0 = n - 2; i1 = n - 1; frac = 1.0;
            return;
        }
        // Binary search for upper bound (equivalent to std::upper_bound)
        int lo = 0, hi = n;
        while (lo < hi) {
            const int mid = (lo + hi) / 2;
            if (grid[mid] <= x) lo = mid + 1;
            else                hi = mid;
        }
        // lo = upper_bound index
        i1 = lo;
        i0 = lo - 1;
        const double denom = grid[i1] - grid[i0];
        frac = (denom > 0.0) ? ((x - grid[i0]) / denom) : 0.0;
    }

    __device__ __forceinline__ static bool interpolate_linear_valid(
        double frac, bool v0_valid, double v0, bool v1_valid, double v1, double& out)
    {
        if (v0_valid && v1_valid) {
            out = v0 + frac * (v1 - v0);
            return true;
        }
        if (v0_valid) {
            out = v0;
            return true;
        }
        if (v1_valid) {
            out = v1;
            return true;
        }
        out = 0.0;
        return false;
    }

    /// Bilinear interpolation of kappa_table[g] at (log_ni, log_T) in log-kappa space.
    /// Exactly replicates host IonmixOpacityData::interpolate_kappa().
    __device__ double interpolate(
        const double* kappa_table, int g,
        double log_ni, double log_T) const
    {
        // Clamp to table bounds in log-space
        const double lni = fmin(fmax(log_ni, log_ni_min), log_ni_max);
        const double lt  = fmin(fmax(log_T,  log_T_min),  log_T_max);

        // Bracket in density and temperature axes
        int d0, d1, t0, t1;
        double tx, ty;
        bracket(log_numdens, ndens, lni, d0, d1, tx);
        bracket(log_temps,   ntemp, lt,  t0, t1, ty);

        // Fetch 4 corner values
        const double k00 = kappa_table[flat_index(g, d0, t0)];
        const double k10 = kappa_table[flat_index(g, d1, t0)];
        const double k01 = kappa_table[flat_index(g, d0, t1)];
        const double k11 = kappa_table[flat_index(g, d1, t1)];

        const bool k00_valid = k00 > kFloorAwareThreshold;
        const bool k10_valid = k10 > kFloorAwareThreshold;
        const bool k01_valid = k01 > kFloorAwareThreshold;
        const bool k11_valid = k11 > kFloorAwareThreshold;

        if (!(k00_valid && k10_valid && k01_valid && k11_valid)) {
            double kx0 = 0.0;
            double kx1 = 0.0;
            double kxy = 0.0;
            const bool kx0_valid =
                interpolate_linear_valid(tx, k00_valid, k00, k10_valid, k10, kx0);
            const bool kx1_valid =
                interpolate_linear_valid(tx, k01_valid, k01, k11_valid, k11, kx1);
            const bool kxy_valid =
                interpolate_linear_valid(ty, kx0_valid, kx0, kx1_valid, kx1, kxy);
            return kxy_valid ? kxy : 0.0;
        }

        // Log-space interpolation (floor at 1e-100 to avoid log(0))
        const double lk00 = log(fmax(k00, kKappaFloor));
        const double lk10 = log(fmax(k10, kKappaFloor));
        const double lk01 = log(fmax(k01, kKappaFloor));
        const double lk11 = log(fmax(k11, kKappaFloor));

        // Bilinear lerp in log-kappa space
        const double lkx0 = lk00 + tx * (lk10 - lk00);
        const double lkx1 = lk01 + tx * (lk11 - lk01);
        const double lk   = lkx0 + ty * (lkx1 - lkx0);

        return exp(lk);
    }
};

}  // namespace tenryu::materials
