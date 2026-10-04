// Serial screening only. Exact narrow-phase arithmetic stays in ws3_pairs.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

using I = int64_t;

extern "C" void ws3_window_screen(
    I frames, I geoms, I pairs, const I* first, const I* second,
    const double* bounds, const double* vertex_radius, double threshold,
    const double* position, const double* rotation,
    const double* old_position, const double* old_rotation,
    const double* old_gap, const uint8_t* old_status, const uint8_t* valid,
    uint8_t* near, uint8_t* reuse, uint8_t* certified, double* gap) {
    constexpr double eps = std::numeric_limits<double>::epsilon();
    std::vector<double> travel(geoms);
    std::vector<uint8_t> same(geoms);
    for (I frame = 0; frame < frames; ++frame) {
        if (valid[frame]) {
            for (I geom = 0; geom < geoms; ++geom) {
                I base = frame * geoms + geom;
                const double* p = position + 3 * base;
                const double* op = old_position + 3 * base;
                const double* r = rotation + 9 * base;
                const double* ore = old_rotation + 9 * base;
                // Byte equality preserves signed zero and exact transform keys.
                same[geom] = !std::memcmp(p, op, 3 * sizeof(double))
                          && !std::memcmp(r, ore, 9 * sizeof(double));
                double translation = 0., turn = 0.;
                for (int k = 0; k < 3; ++k) {
                    double delta = p[k] - op[k];
                    translation += delta * delta;
                }
                for (int k = 0; k < 9; ++k) {
                    double delta = r[k] - ore[k];
                    turn += delta * delta;
                }
                // For every local vertex v, |delta t + delta R v| is at most
                // |delta t| + ||delta R||_F max|v|, including both bodies.
                travel[geom] = std::sqrt(translation)
                             + std::sqrt(turn) * vertex_radius[geom];
            }
        }
        for (I pair = 0; pair < pairs; ++pair) {
            I index = frame * pairs + pair, a = first[pair], b = second[pair];
            bool bounded = bounds[a] > 0 && bounds[b] > 0;
            near[index] = reuse[index] = certified[index] = 0;
            if (valid[frame] && bounded) {
                double moved = travel[a] + travel[b];
                double slack = 1e-12 + 128 * eps * (std::abs(old_gap[index])
                    + moved + bounds[a] + bounds[b] + std::abs(threshold) + 1.);
                double lower = old_gap[index] - moved;
                // Reverse triangle inequality for the ORIGINAL sphere test.
                // Do not screen using a mesh gap: its first-plane answer at a
                // new pose need not attain a tighter physical distance bound.
                if (std::isfinite(lower) && lower > threshold + slack) {
                    gap[index] = lower;
                    certified[index] = 1;
                    continue;
                }
            }
            const double* pa = position + 3 * (frame * geoms + a);
            const double* pb = position + 3 * (frame * geoms + b);
            double norm2 = 0.;
            for (int k = 0; k < 3; ++k) {
                double delta = pa[k] - pb[k];
                norm2 += delta * delta;
            }
            double distance = std::sqrt(norm2);
            gap[index] = distance - bounds[a] - bounds[b];
            double cutoff = bounds[a] + bounds[b] + threshold;
            double slack = 1e-12 + 128 * eps * (distance + std::abs(cutoff) + 1.);
            near[index] = !bounded || distance <= cutoff;
            // Re-evaluate exact-threshold ties with the original NumPy norm.
            if (bounded && std::abs(distance - cutoff) <= slack) near[index] = 2;
            reuse[index] = valid[frame] && same[a] && same[b]
                         && (old_status[index] == 1 || old_status[index] == 3);
        }
    }
}
