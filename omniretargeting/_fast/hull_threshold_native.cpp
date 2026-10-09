// Boolean hull tests only. Unresolved roundoff cases keep the original BLAS path.
#include <cmath>
#include <cstdint>
#include <limits>
#include <algorithm>

extern "C" void hull_thresholds(int64_t n, const double* points, int64_t faces,
    const double* planes, int64_t bands, const double* thresholds, int8_t* output) {
  double largest = -std::numeric_limits<double>::infinity();
  for (int64_t k = 0; k < bands; ++k) largest = std::max(largest, thresholds[k]);
  for (int64_t i = 0; i < n; ++i) {
    double lower = -std::numeric_limits<double>::infinity(), upper = lower;
    bool finite = true;
    for (int64_t j = 0; j < faces; ++j) {
      const double* p = planes + 4*j;
      double a = points[3*i]*p[0], b = points[3*i+1]*p[1], c = points[3*i+2]*p[2];
      double value = ((a + b) + c) + p[3];
      // Covers product/sum rounding in both this loop and NumPy's BLAS,
      // including cancellation and underflow. This is a decision certificate,
      // never a distance, witness, normal or winding approximation.
      double error = 1e-12 * (1. + std::abs(a) + std::abs(b) + std::abs(c) + std::abs(p[3]));
      if (!std::isfinite(value) || !std::isfinite(error)) { finite = false; break; }
      lower = std::max(lower, value - error);
      upper = std::max(upper, value + error);
      if (lower > largest) break;
    }
    for (int64_t k = 0; k < bands; ++k)
      output[i*bands+k] = !finite ? -1 : lower > thresholds[k] ? 1 : upper <= thresholds[k] ? 0 : -1;
  }
}
