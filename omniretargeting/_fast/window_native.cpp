// Original double arithmetic, SciPy corner order, float32 grid result.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

static double lookup(const float* grid, const double* lower, double spacing,
                     const int64_t* shape, const double* point) {
    double t[3]; int64_t base[3];
    for (int axis = 0; axis < 3; ++axis) {
        double x = (point[axis] - lower[axis]) / spacing;
        if (!(x >= 0 && x <= shape[axis] - 1))
            return std::numeric_limits<double>::quiet_NaN();
        base[axis] = int64_t(std::floor(x)); t[axis] = x - base[axis];
    }
    double total = 0;
    for (int corner = 0; corner < 8; ++corner) {
        int dx = (corner >> 2) & 1, dy = (corner >> 1) & 1, dz = corner & 1;
        auto x = std::min(base[0] + dx, shape[0] - 1);
        auto y = std::min(base[1] + dy, shape[1] - 1);
        auto z = std::min(base[2] + dz, shape[2] - 1);
        double value = grid[(x * shape[1] + y) * shape[2] + z];
        value *= dx ? t[0] : 1.0 - t[0];
        value *= dy ? t[1] : 1.0 - t[1];
        value *= dz ? t[2] : 1.0 - t[2];
        total += value;
    }
    return static_cast<float>(total);
}

static void transform(const double* rotation, const double* point, double* out) {
    for (int axis = 0; axis < 3; ++axis)
        out[axis] = (rotation[3 * axis] * point[0] + rotation[3 * axis + 1] * point[1])
                  + rotation[3 * axis + 2] * point[2];
}

static void field_point(const double* rotation, const double* translation,
                        double scale, const double* point, double* out) {
    if (!rotation) {
        std::copy(point, point + 3, out); return;
    }
    double delta[3];
    for (int axis = 0; axis < 3; ++axis) delta[axis] = point[axis] - translation[axis];
    for (int axis = 0; axis < 3; ++axis)
        out[axis] = ((rotation[axis] * delta[0] + rotation[3 + axis] * delta[1])
                   + rotation[6 + axis] * delta[2]) / scale;
}

extern "C" void r3_grid(int64_t count, const float* grid, const double* lower,
                         double spacing, const int64_t* shape,
                         const double* points, float* values) {
    for (int64_t row = 0; row < count; ++row)
        values[row] = lookup(grid, lower, spacing, shape, points + 3 * row);
}

extern "C" void r3_support(int64_t count, int64_t vertices, const double* directions,
    const double* points, double margin, int64_t* indices, uint8_t* uncertain) {
    for (int64_t row = 0; row < count; ++row) {
        const double* d = directions + 3 * row;
        double best = -std::numeric_limits<double>::infinity(), second = best;
        int64_t choice = 0;
        for (int64_t vertex = 0; vertex < vertices; ++vertex) {
            const double* p = points + 3 * vertex;
            double value = (d[0] * p[0] + d[1] * p[1]) + d[2] * p[2];
            if (value > best) { second = best; best = value; choice = vertex; }
            else if (value > second) second = value;
        }
        indices[row] = choice;
        uncertain[row] = !(best - second > margin);
    }
}

extern "C" void r3_shell(int64_t frames, int64_t count, int64_t ngeom,
    int64_t max_samples, const int64_t* geoms, const int64_t* indices,
    const double* local, const double* position, const double* rotation,
    const float* grid, const double* lower, double spacing, const int64_t* shape,
    const double* field_rotation, const double* field_translation,
    const double* scales, const double* threshold,
    double* clearance, int64_t* choices) {
    const double infinity = std::numeric_limits<double>::infinity();
    for (int64_t frame = 0; frame < frames; ++frame) {
        const double* fr = field_rotation ? field_rotation + frame * 9 : nullptr;
        const double* ft = field_translation ? field_translation + frame * 3 : nullptr;
        double scale = scales ? scales[frame] : 1;
        for (int64_t geom = 0; geom < count; ++geom) {
            auto row = frame * count + geom;
            clearance[row] = infinity; choices[row] = -1;
            const double* origin = position + (frame * ngeom + geoms[geom]) * 3;
            const double* r = rotation + (frame * ngeom + geoms[geom]) * 9;
            double p[3]; field_point(fr, ft, scale, origin, p);
            double centre = lookup(grid, lower, spacing, shape, p);
            if (scales) centre *= scale;
            if (!(std::isfinite(centre) && centre < threshold[geom])) continue;
            double best = infinity; int64_t choice = 0;
            for (int64_t sample = 0; sample < max_samples; ++sample) {
                auto index = indices[geom * max_samples + sample];
                if (index < 0) continue;
                double world[3]; transform(r, local + index * 3, world);
                for (int axis = 0; axis < 3; ++axis) world[axis] += origin[axis];
                field_point(fr, ft, scale, world, p);
                double value = lookup(grid, lower, spacing, shape, p);
                if (scales) value *= scale;
                if (std::isfinite(value) && value < best) { best = value; choice = index; }
            }
            clearance[row] = best;
            // Original argmin selects the first sample if all are unknown.
            choices[row] = std::isfinite(best) ? choice : indices[geom * max_samples];
        }
    }
}
