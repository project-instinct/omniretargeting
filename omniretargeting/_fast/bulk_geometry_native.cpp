// Acquisition-only support and shell kernels; reference arithmetic, no FMA.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

extern "C" void bulk_support(int64_t count, int64_t vertices,
                             const double* directions, const double* points,
                             int64_t* choices, double* margins) {
  for (int64_t i = 0; i < count; ++i) {
    const double* d = directions + 3*i;
    double best = -INFINITY, second = -INFINITY;
    int64_t choice = 0;
    for (int64_t j = 0; j < vertices; ++j) {
      const double* p = points + 3*j;
      double v = (d[0]*p[0] + d[1]*p[1]) + d[2]*p[2];
      if (v > best) { second = best; best = v; choice = j; }
      else if (v > second) second = v;
    }
    choices[i] = choice;
    margins[i] = best - second;
  }
}

// SciPy ndimage order=1: x/y/z products, z-fastest corner traversal,
// including unknown corners with zero weight and float32 output rounding.
static float lookup(const float* grid, const int64_t* shape,
                    const double* lower, double spacing, const double* point) {
  int64_t base[3]; double t[3];
  for (int a = 0; a < 3; ++a) {
    double x = (point[a] - lower[a]) / spacing;
    if (!(x >= 0 && x <= shape[a]-1)) return NAN;
    double f = std::floor(x);
    base[a] = static_cast<int64_t>(f); t[a] = x - f;
  }
  double total = 0;
  for (int c = 0; c < 8; ++c) {
    int dx = (c >> 2)&1, dy = (c >> 1)&1, dz = c&1;
    int64_t x = std::min(base[0]+dx, shape[0]-1);
    int64_t y = std::min(base[1]+dy, shape[1]-1);
    int64_t z = std::min(base[2]+dz, shape[2]-1);
    double v = grid[(x*shape[1]+y)*shape[2]+z];
    v *= dx ? t[0] : 1-t[0];
    v *= dy ? t[1] : 1-t[1];
    v *= dz ? t[2] : 1-t[2];
    total += v;
  }
  return static_cast<float>(total);
}

extern "C" void bulk_grid(int64_t count, const float* grid,
                          const int64_t* shape, const double* lower,
                          double spacing, const double* points, float* values) {
  for (int64_t i = 0; i < count; ++i)
    values[i] = lookup(grid, shape, lower, spacing, points+3*i);
}

extern "C" void bulk_shell(int64_t count, int64_t samples, int64_t geom,
                           int64_t ngeom, const int64_t* frames,
                           const double* local, const double* position,
                           const double* rotation, const double* field_rotation,
                           const double* translation, const double* scales,
                           const float* grid, const int64_t* shape,
                           const double* lower, double spacing,
                           double* deepest, int64_t* choices) {
  for (int64_t i = 0; i < count; ++i) {
    int64_t f = frames[i];
    const double* p = position + (f*ngeom+geom)*3;
    const double* r = rotation + (f*ngeom+geom)*9;
    double best = INFINITY;
    int64_t choice = 0;
    double scale = scales ? scales[f] : 1;
    for (int64_t j = 0; j < samples; ++j) {
      const double* v = local + 3*j;
      double world[3], field[3];
      for (int a = 0; a < 3; ++a)
        world[a] = p[a] + ((r[3*a]*v[0]+r[3*a+1]*v[1])+r[3*a+2]*v[2]);
      if (field_rotation) {
        const double* fr = field_rotation+9*f;
        double u[3];
        for (int a = 0; a < 3; ++a) u[a] = world[a]-translation[3*f+a];
        for (int a = 0; a < 3; ++a)
          field[a] = ((fr[a]*u[0]+fr[3+a]*u[1])+fr[6+a]*u[2]) / scale;
      } else std::copy(world, world+3, field);
      double value = static_cast<double>(lookup(grid, shape, lower, spacing, field))*scale;
      if (std::isfinite(value) && value < best) { best = value; choice = j; }
    }
    deepest[i] = best; choices[i] = choice;
  }
}
