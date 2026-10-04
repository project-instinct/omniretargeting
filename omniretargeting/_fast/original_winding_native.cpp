// Persistent libigl/SideFX winding BVHs. Queries are strictly serial.
#include <vector>
#include <cstdint>
#include <cmath>
#include <cassert>
#include <algorithm>
#include "FastWindingNumberForSoups.h"

using Vec = igl::FastWindingNumber::HDK_Sample::UT_Vector3T<float>;
using Solid = igl::FastWindingNumber::HDK_Sample::UT_SolidAngle<float, float>;
struct State {
  std::vector<Vec> vertices;
  std::vector<int> faces;
  Solid winding;
};
extern "C" void* r4_winding_build(int64_t nv, const double* vertices,
                                  int64_t nf, const int64_t* faces) {
  try {
    auto* s = new State;
    s->vertices.resize(nv);
    s->faces.resize(nf * 3);
    for (int64_t i = 0; i < nv; ++i)
      for (int j = 0; j < 3; ++j) s->vertices[i][j] = vertices[i * 3 + j];
    for (int64_t i = 0; i < nf * 3; ++i) s->faces[i] = faces[i];
    s->winding.init(nf, s->faces.data(), nv, s->vertices.data(), 2);
    return s;
  } catch (...) { return nullptr; }
}
extern "C" void r4_winding_query(void* state, int64_t n, const double* points, double* result) {
  const auto* s = static_cast<State*>(state);
  constexpr double four_pi = 4.0 * 3.1415926535897932384626433832795;
  for (int64_t i = 0; i < n; ++i) {
    Vec p;
    for (int j = 0; j < 3; ++j) p[j] = points[i * 3 + j];
    result[i] = s->winding.computeSolidAngle(p, 2.0f) / four_pi;
  }
}
extern "C" void r4_winding_free(void* state) { delete static_cast<State*>(state); }

extern "C" void r4_grid_lower(int64_t n, const double* points, const double* origin,
                               double spacing, const int64_t* shape,
                               const double* values, double* result) {
  for (int64_t i = 0; i < n; ++i) {
    int64_t base[3];
    bool inside = true;
    for (int j = 0; j < 3; ++j) {
      const double index = (points[3*i+j] - origin[j]) / spacing;
      inside &= std::isfinite(index) && index >= 0 && index <= shape[j]-1;
      base[j] = std::isfinite(index) ? std::min<int64_t>(std::max<int64_t>(0, std::floor(index)), shape[j]-2) : 0;
    }
    double lower = -INFINITY;
    if (inside) for (int corner = 0; corner < 8; ++corner) {
      int64_t node[3];
      double distance2 = 0;
      for (int j = 0; j < 3; ++j) {
        node[j] = base[j] + ((corner >> j) & 1);
        const double delta = points[3*i+j] - (origin[j] + spacing*node[j]);
        distance2 += delta*delta;
      }
      const double value = values[(node[0]*shape[1] + node[1])*shape[2] + node[2]];
      if (std::isfinite(value)) lower = std::max(lower, value - std::sqrt(distance2) - 1e-9);
    }
    result[i] = lower;
  }
}
