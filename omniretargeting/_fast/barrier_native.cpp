// Scatter weighted CSR row outer products directly into lower band storage.
// No sparse normal matrix, no frame/coordinate reordering, FP64 throughout.
#include <cstdint>
extern "C" int barrier_scatter(int64_t rows, int64_t band, int64_t size,
    const int64_t* pointer, const int64_t* column, const double* data,
    const double* weight, double* ab) {
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t i = pointer[r]; i < pointer[r+1]; ++i) {
      const int64_t ci = column[i];
      for (int64_t j = pointer[r]; j <= i; ++j) {
        const int64_t cj = column[j], offset = ci-cj;
        if (offset < 0 || offset >= band || ci >= size) return 1;
        ab[offset + band*cj] += weight[r]*data[i]*data[j];
      }
    }
  }
  return 0;
}
