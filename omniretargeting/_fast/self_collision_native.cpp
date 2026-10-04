// Batched version of convex_mesh_distance.py's *same* Qhull expansion.
// Arithmetic uses NumPy's BLAS, including its support-query reduction order.
// Penetration witnesses use NumPy's LAPACK; failures use the original routine.
#include "_qhull_r/qhull_ra.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using I = int64_t;
using Gemm = void (*)(int, int, int, I, I, I, double, const double*, I,
                     const double*, I, double, double*, I);
using Gemv = void (*)(int, int, I, I, double, const double*, I,
                     const double*, I, double, double*, I);
using Dot = double (*)(I, const double*, I, const double*, I);
using Gelsd = void (*)(I*, I*, I*, double*, I*, double*, I*, double*,
                      double*, I*, double*, I*, I*, I*);
static Gemm gemm;
static Gemv gemv;
static Dot dot;
static Gelsd gelsd;

// Batch the original MuJoCo calls, copying the same buffers in frame order.
extern "C" void ws3_fk(
    I frames, I bodies, I geoms, I nq, I nv, I with_jacobian,
    void* kinematics_function, void* com_function, void* jac_function,
    const void* model, void* data, const double* poses, double* qpos,
    const double* body_position, const double* body_rotation,
    const double* geom_position, const double* geom_rotation,
    double* positions, double* rotations, double* geom_positions,
    double* geom_rotations, double* jacp, double* jacr) {
    using Stage = void (*)(const void*, void*);
    using Jac = void (*)(const void*, const void*, double*, double*, int);
    Stage kinematics = reinterpret_cast<Stage>(kinematics_function);
    Stage com = reinterpret_cast<Stage>(com_function);
    Jac jac = reinterpret_cast<Jac>(jac_function);
    for (I frame = 0; frame < frames; ++frame) {
        std::memcpy(qpos, poses + frame * nq, nq * sizeof(double));
        kinematics(model, data);
        std::memcpy(positions + frame * bodies * 3, body_position, bodies * 3 * sizeof(double));
        std::memcpy(rotations + frame * bodies * 9, body_rotation, bodies * 9 * sizeof(double));
        std::memcpy(geom_positions + frame * geoms * 3, geom_position, geoms * 3 * sizeof(double));
        std::memcpy(geom_rotations + frame * geoms * 9, geom_rotation, geoms * 9 * sizeof(double));
        if (!with_jacobian) continue;
        com(model, data);
        for (I body = 1; body < bodies; ++body) {
            I offset = (frame * bodies + body) * 3 * nv;
            jac(model, data, jacp + offset, jacr + offset, int(body));
        }
    }
}

// The same un-fused arithmetic as kinematics.cross_columns and NumPy's
// three-component einsum. Avoid large temporary translational Jacobians.
extern "C" void ws3_point_rows(
    I frames, I bodies, I points, I width, const I* ids,
    const double* position, const double* origins, const double* jacp,
    const double* jacr, const double* directions, double* result) {
    for (I frame = 0; frame < frames; ++frame) {
        for (I point = 0; point < points; ++point) {
            I row = frame * points + point, body = frame * bodies + ids[row];
            double x = position[3 * row] - origins[3 * body];
            double y = position[3 * row + 1] - origins[3 * body + 1];
            double z = position[3 * row + 2] - origins[3 * body + 2];
            const double* p = jacp + body * 3 * width;
            const double* r = jacr + body * 3 * width;
            double dx = directions[3 * row], dy = directions[3 * row + 1], dz = directions[3 * row + 2];
            for (I column = 0; column < width; ++column) {
                double a = p[column] - (y * r[2 * width + column] - z * r[width + column]);
                double b = p[width + column] - (z * r[column] - x * r[2 * width + column]);
                double c = p[2 * width + column] - (x * r[width + column] - y * r[column]);
                result[row * width + column] = (dx * a + dy * b) + dz * c;
            }
        }
    }
}

extern "C" void ws3_point_jac(
    I frames, I bodies, I points, I width, const I* ids,
    const double* position, const double* origins, const double* jacp,
    const double* jacr, double* result) {
    for (I frame = 0; frame < frames; ++frame) {
        for (I point = 0; point < points; ++point) {
            I row = frame * points + point, body = frame * bodies + ids[row];
            double x = position[3 * row] - origins[3 * body];
            double y = position[3 * row + 1] - origins[3 * body + 1];
            double z = position[3 * row + 2] - origins[3 * body + 2];
            const double* p = jacp + body * 3 * width;
            const double* r = jacr + body * 3 * width;
            double* out = result + row * 3 * width;
            for (I column = 0; column < width; ++column) {
                out[column] = p[column] - (y * r[2 * width + column] - z * r[width + column]);
                out[width + column] = p[width + column] - (z * r[column] - x * r[2 * width + column]);
                out[2 * width + column] = p[2 * width + column] - (x * r[width + column] - y * r[column]);
            }
        }
    }
}

extern "C" void ws3_blas(void* mm, void* mv, void* dd, void* ls) {
    gemm = reinterpret_cast<Gemm>(mm);
    gemv = reinterpret_cast<Gemv>(mv);
    dot = reinterpret_cast<Dot>(dd);
    gelsd = reinterpret_cast<Gelsd>(ls);
}

// Call MuJoCo's original point Jacobian, then NumPy's original GEMV. Moving
// the per-witness loop out of Python saves allocations and binding calls;
// it does not replace mj_jac with a rearranged body-Jacobian expression.
extern "C" void ws3_collision_jac(
    I count, I width, void* jac_function, const void* model, const void* data,
    const I* bodies, const double* points, const double* directions,
    double* result) {
    using Jac = void (*)(const void*, const void*, double*, double*, const double*, int);
    Jac jac = reinterpret_cast<Jac>(jac_function);
    std::vector<double> first(3 * width), second(3 * width);
    for (I row = 0; row < count; ++row) {
        std::fill(first.begin(), first.end(), 0.);
        jac(model, data, first.data(), nullptr, points + 6 * row, int(bodies[2 * row]));
        if (bodies[2 * row + 1]) {
            std::fill(second.begin(), second.end(), 0.);
            jac(model, data, second.data(), nullptr, points + 6 * row + 3,
                int(bodies[2 * row + 1]));
            for (I i = 0; i < 3 * width; ++i) first[i] -= second[i];
        }
        gemv(101, 112, 3, width, 1., first.data(), width, directions + 3 * row,
             1, 0., result + row * width, 1);
    }
}

struct Hull {
    qhT state;
    qhT* qh = &state;
    bool initialized = false;
    Hull(double* points, int count, FILE* errors) {
        qh_zero(qh, errors);
        initialized = true;
        char options[] = "qhull i Qt";
        int status = qh_new_qhull(qh, 3, count, points, 0, options, nullptr, errors);
        if (status) return;
        qh_triangulate(qh);
        valid = true;
    }
    bool valid = false;
    ~Hull() {
        if (!initialized) return;
        int curlong, totlong;
        qh_freeqhull(qh, !qh_ALL);
        qh_memfreeshort(qh, &curlong, &totlong);
    }
    facetT* first() { return qh->facet_list; }
};

// A startup check compares these equations, including facet order, to SciPy.
extern "C" int ws3_hull(double* points, int count, double* equations) {
    FILE* errors = fopen("/dev/null", "w");
    int result = -1;
    {
        Hull h(points, count, errors);
        if (h.valid) {
            result = 0;
            for (facetT* f = h.first(); f && f->next; f = f->next) {
                for (int k = 0; k < 3; ++k) equations[4 * result + k] = f->normal[k];
                equations[4 * result + 3] = f->offset;
                ++result;
            }
        }
    }
    fclose(errors);
    return result;
}

struct World {
    std::vector<double> points;
    double mean[3];
    double maximum[78], minimum[78], max_margin[26], min_margin[26];
    double magnitude = 0.;
    bool seeds_ready = false;
};

static void prepare(World& w, const double* local, I count,
                    const double* rotation, const double* position) {
    w.points.resize(count * 3);
    gemm(101, 111, 112, count, 3, 3, 1., local, 3, rotation, 3,
         0., w.points.data(), 3);
    for (int k = 0; k < 3; ++k) w.mean[k] = 0.;
    for (I i = 0; i < count; ++i) {
        for (int k = 0; k < 3; ++k) {
            double& p = w.points[3 * i + k];
            p += position[k];
            w.mean[k] += p;
            w.magnitude = std::max(w.magnitude, std::abs(p));
        }
    }
    for (int k = 0; k < 3; ++k) w.mean[k] /= count;
}

static void prepare_seeds(World& w, const double* seed) {
    if (w.seeds_ready) return;
    I count = w.points.size() / 3;
    // These scores only certify unique indices, never become an output.
    // Opposite axes share extrema. Retain all axes for an asymmetric seed.
    bool symmetric = true;
    for (int axis = 0; axis < 13; ++axis)
        for (int k = 0; k < 3; ++k)
            symmetric &= seed[3 * (25 - axis) + k] == -seed[3 * axis + k];
    int axes = symmetric ? 13 : 26;
    for (int axis = 0; axis < axes; ++axis) {
        I high = 0, low = 0;
        double highest = -INFINITY, lowest = INFINITY;
        double second_high = -INFINITY, second_low = INFINITY;
        for (I i = 0; i < count; ++i) {
            const double* p = w.points.data() + i * 3;
            const double* n = seed + axis * 3;
            double value = (p[0] * n[0] + p[1] * n[1]) + p[2] * n[2];
            if (value > highest) { second_high = highest; highest = value; high = i; }
            else second_high = std::max(second_high, value);
            if (value < lowest) { second_low = lowest; lowest = value; low = i; }
            else second_low = std::min(second_low, value);
        }
        w.max_margin[axis] = highest - second_high;
        w.min_margin[axis] = second_low - lowest;
        for (int k = 0; k < 3; ++k) {
            w.maximum[axis * 3 + k] = w.points[high * 3 + k];
            w.minimum[axis * 3 + k] = w.points[low * 3 + k];
        }
        if (symmetric) {
            int opposite = 25 - axis;
            w.max_margin[opposite] = w.min_margin[axis];
            w.min_margin[opposite] = w.max_margin[axis];
            for (int k = 0; k < 3; ++k) {
                w.maximum[opposite * 3 + k] = w.minimum[axis * 3 + k];
                w.minimum[opposite * 3 + k] = w.maximum[axis * 3 + k];
            }
        }
    }
    w.seeds_ready = true;
}

static I extreme(const std::vector<double>& scores, I count, I stride, I axis,
                 bool maximum) {
    I choice = 0;
    double value = scores[axis];
    for (I i = 1; i < count; ++i) {
        double candidate = scores[i * stride + axis];
        if (maximum ? candidate > value : candidate < value) {
            choice = i;
            value = candidate;
        }
    }
    return choice;
}

static void support(const std::vector<double>& a, const std::vector<double>& b,
                    const double* axes, I n, std::vector<double>& ascores,
                    std::vector<double>& bscores, double* pa, double* pb) {
    I na = a.size() / 3, nb = b.size() / 3;
    ascores.resize(na * n); bscores.resize(nb * n);
    if (n == 1) {
        gemv(101, 111, na, 3, 1., a.data(), 3, axes, 1, 0., ascores.data(), 1);
        gemv(101, 111, nb, 3, 1., b.data(), 3, axes, 1, 0., bscores.data(), 1);
    } else {
        gemm(101, 111, 112, na, n, 3, 1., a.data(), 3, axes, 3,
             0., ascores.data(), n);
        gemm(101, 111, 112, nb, n, 3, 1., b.data(), 3, axes, 3,
             0., bscores.data(), n);
    }
    for (I k = 0; k < n; ++k) {
        I ai = extreme(ascores, na, n, k, true);
        I bi = extreme(bscores, nb, n, k, false);
        for (int j = 0; j < 3; ++j) {
            pa[3 * k + j] = a[3 * ai + j];
            pb[3 * k + j] = b[3 * bi + j];
        }
    }
}

// Materialize pair-centered coordinates only for uncertified support choices.
struct Centered {
    World& world;
    const double* origin;
    std::vector<double> points;
    void prepare() {
        if (!points.empty()) return;
        points.resize(world.points.size());
        for (I i = 0; i < I(points.size()); ++i)
            points[i] = world.points[i] - origin[i % 3];
    }
};

static void one_support(Centered& centered, const double* axis, bool maximum,
                        std::vector<double>& scores, double* point) {
    World& w = centered.world;
    I count = w.points.size() / 3;
    scores.resize(count);
    I choice;
    if (!centered.points.empty()) {
        gemv(101, 111, count, 3, 1., centered.points.data(), 3,
             axis, 1, 0., scores.data(), 1);
        choice = extreme(scores, count, 1, 0, maximum);
    } else {
        gemv(101, 111, count, 3, 1., w.points.data(), 3,
             axis, 1, 0., scores.data(), 1);
        choice = 0;
        double best = scores[0], second = maximum ? -INFINITY : INFINITY;
        for (I i = 1; i < count; ++i) {
            double value = scores[i];
            if (maximum ? value > best : value < best) {
                second = best; best = value; choice = i;
            } else second = maximum ? std::max(second, value) : std::min(second, value);
        }
        double origin_size = std::max({std::abs(centered.origin[0]),
            std::abs(centered.origin[1]), std::abs(centered.origin[2])});
        double norm = std::abs(axis[0]) + std::abs(axis[1]) + std::abs(axis[2]);
        // For each vertex, subtracting the origin and either three-term dot
        // product changes its ideal score by at most a few eps times
        // (max|world point| + max|origin|) * ||axis||_1. This slack exceeds
        // twice that bound, so a unique winner remains the reference winner.
        // Ties, small margins and nonfinite bounds use the original GEMV.
        double slack = 128 * 2.2204460492503131e-16
                     * (w.magnitude + origin_size + 1.) * norm;
        double margin = maximum ? best - second : second - best;
        if (!(margin > slack)) {
            centered.prepare();
            gemv(101, 111, count, 3, 1., centered.points.data(), 3,
                 axis, 1, 0., scores.data(), 1);
            choice = extreme(scores, count, 1, 0, maximum);
        }
    }
    for (int k = 0; k < 3; ++k)
        point[k] = w.points[choice * 3 + k] - centered.origin[k];
}

// Return 1 for an exact separating answer, 2 for the original narrow phase.
static uint8_t distance(World& wa, World& wb, const double* seed,
                        FILE* errors, double& clearance, double* witnesses,
                        double* direction, const I* choices_a,
                        const I* choices_b, const uint8_t* reliable_a,
                        const uint8_t* reliable_b) {
    double origin[3];
    for (int k = 0; k < 3; ++k) origin[k] = (wa.mean[k] + wb.mean[k]) / 2.;
    Centered a{wa, origin, {}}, b{wb, origin, {}};
    std::vector<double> ascores, bscores;
    double pa[3 * 154], pb[3 * 154], vertices[3 * 154];
    bool certified = choices_a && choices_b && reliable_a && reliable_b;
    if (certified) {
        for (int axis = 0; axis < 26; ++axis)
            certified &= reliable_a[axis] && reliable_b[26 + axis];
    }
    if (certified) {
        for (int axis = 0; axis < 26; ++axis) {
            for (int k = 0; k < 3; ++k) {
                pa[3 * axis + k] = wa.points[3 * choices_a[axis] + k] - origin[k];
                pb[3 * axis + k] = wb.points[3 * choices_b[26 + axis] + k] - origin[k];
            }
        }
    } else {
        prepare_seeds(wa, seed); prepare_seeds(wb, seed);
        double origin_size = std::max({std::abs(origin[0]), std::abs(origin[1]), std::abs(origin[2])});
        // Recentring shifts every exact score by the same amount. Only reuse a
        // unique support vertex when its score margin exceeds the error of both
        // three-term floating dot products and coordinate subtraction. Ties and
        // near ties evaluate the reference's original recentered GEMM instead.
        double a_slack = 64 * 2.2204460492503131e-16 * (wa.magnitude + origin_size + 1.);
        double b_slack = 64 * 2.2204460492503131e-16 * (wb.magnitude + origin_size + 1.);
        bool ambiguous = false;
        for (int axis = 0; axis < 26; ++axis)
            ambiguous |= wa.max_margin[axis] <= a_slack || wb.min_margin[axis] <= b_slack;
        if (ambiguous) {
            a.prepare(); b.prepare();
            support(a.points, b.points, seed, 26, ascores, bscores, pa, pb);
        }
        else {
            for (int i = 0; i < 78; ++i) {
                pa[i] = wa.maximum[i] - origin[i % 3];
                pb[i] = wb.minimum[i] - origin[i % 3];
            }
        }
    }
    for (int i = 0; i < 78; ++i) vertices[i] = pa[i] - pb[i];
    int count = 26;
    for (int iteration = 0; iteration < 128; ++iteration) {
        Hull hull(vertices, count, errors);
        if (!hull.valid) return 2;
        facetT* face = hull.first();
        for (facetT* f = face; f && f->next; f = f->next) {
            if (f->offset > face->offset) face = f;
        }
        double normal[3];
        for (int k = 0; k < 3; ++k) normal[k] = face->normal[k];
        double sa[3], sb[3], point[3];
        one_support(a, normal, true, ascores, sa);
        one_support(b, normal, false, bscores, sb);
        for (int k = 0; k < 3; ++k) point[k] = sa[k] - sb[k];
        double upper = dot(3, point, 1, normal, 1);
        if (upper <= 1e-9) {
            clearance = std::max(0., -upper);
            for (int k = 0; k < 3; ++k) {
                witnesses[k] = sa[k] + origin[k];
                witnesses[3 + k] = sb[k] + origin[k];
                direction[k] = -normal[k];
            }
            return 1;
        }
        double lower = -face->offset;
        if (lower >= 0 && upper - lower <= 1e-9) {
            // NumPy lstsq uses DGELSD, m=4, n=3, and rcond=max(m,n)*eps.
            // Keep the reference's facet order and first feasible triangle.
            std::vector<double> equations;
            std::vector<facetT*> faces;
            for (facetT* f = hull.first(); f && f->next; f = f->next) {
                faces.push_back(f);
                for (int k = 0; k < 3; ++k) equations.push_back(f->normal[k]);
                equations.push_back(f->offset);
            }
            std::vector<double> alignment(faces.size());
            gemv(101, 111, faces.size(), 3, 1., equations.data(), 4,
                 normal, 1, 0., alignment.data(), 1);
            for (I fi = 0; fi < I(faces.size()); ++fi) {
                if (!(alignment[fi] > 1 - 1e-8 &&
                      std::abs(faces[fi]->offset + lower) < 1e-8)) continue;
                I ids[3];
                for (int k = 0; k < 3; ++k) {
                    vertexT* v = reinterpret_cast<vertexT*>(faces[fi]->vertices->e[k].p);
                    ids[k] = qh_pointid(hull.qh, v->point);
                }
                double matrix[12], rhs[4], singular[3];
                for (int k = 0; k < 3; ++k) {
                    for (int axis = 0; axis < 3; ++axis)
                        matrix[4 * k + axis] = vertices[3 * ids[k] + axis];
                    matrix[4 * k + 3] = 1.;
                    rhs[k] = lower * normal[k];
                }
                rhs[3] = 1.;
                I m = 4, n = 3, nrhs = 1, lda = 4, ldb = 4, rank, info;
                double rcond = 4 * 2.2204460492503131e-16;
                // Larger than the workspace query for this fixed 4x3 SVD.
                I lwork = 4096, iwork[256];
                double work[4096];
                gelsd(&m, &n, &nrhs, matrix, &lda, rhs, &ldb, singular,
                      &rcond, &rank, work, &lwork, iwork, &info);
                if (info) return 2;
                if (std::min({rhs[0], rhs[1], rhs[2]}) < -1e-6) continue;
                double triangle_a[9], triangle_b[9];
                for (int k = 0; k < 3; ++k) {
                    for (int axis = 0; axis < 3; ++axis) {
                        triangle_a[3 * k + axis] = pa[3 * ids[k] + axis];
                        triangle_b[3 * k + axis] = pb[3 * ids[k] + axis];
                    }
                }
                gemv(101, 112, 3, 3, 1., triangle_a, 3, rhs, 1, 0., witnesses, 1);
                gemv(101, 112, 3, 3, 1., triangle_b, 3, rhs, 1, 0., witnesses + 3, 1);
                for (int k = 0; k < 3; ++k) {
                    witnesses[k] += origin[k]; witnesses[3 + k] += origin[k];
                    direction[k] = -normal[k];
                }
                clearance = -lower;
                return 1;
            }
            return 2;
        }
        for (int i = 0; i < count; ++i) {
            double norm2 = 0.;
            for (int k = 0; k < 3; ++k) {
                double d = vertices[3 * i + k] - point[k];
                norm2 += d * d;
            }
            // The reference compares sqrt(sum(d*d)) to 1e-12. Resolve the
            // rounding band around its squared cutoff with that routine.
            if (norm2 <= 1e-24 * (1 + 64 * 2.2204460492503131e-16)) return 2;
        }
        for (int k = 0; k < 3; ++k) {
            pa[3 * count + k] = sa[k]; pb[3 * count + k] = sb[k];
            vertices[3 * count + k] = point[k];
        }
        ++count;
    }
    return 2;
}

extern "C" void ws3_pairs(
    I frames, I geoms, I pairs, const I* first, const I* second,
    const I* starts, const I* counts, const double* local,
    const double* rotations, const double* positions, const uint8_t* near,
    const double* seed, double* clearance, double* witnesses,
    double* direction, uint8_t* status, const I* choices,
    const uint8_t* reliable) {
    FILE* errors = fopen("/dev/null", "w");
    for (I frame = 0; frame < frames; ++frame) {
        std::vector<World> world(geoms);
        for (I pair = 0; pair < pairs; ++pair) {
            I index = frame * pairs + pair;
            if (!near[index]) continue;
            I a = first[pair], b = second[pair];
            for (I geom : {a, b}) {
                if (!counts[geom]) continue;
                if (world[geom].points.empty())
                    prepare(world[geom], local + starts[geom] * 3, counts[geom],
                            rotations + (frame * geoms + geom) * 9,
                            positions + (frame * geoms + geom) * 3);
            }
            if (!counts[a] || !counts[b]) { status[index] = 2; continue; }
            status[index] = distance(world[a], world[b], seed, errors,
                                     clearance[index], witnesses + index * 6,
                                     direction + index * 3,
                                     choices ? choices + (frame * geoms + a) * 52 : nullptr,
                                     choices ? choices + (frame * geoms + b) * 52 : nullptr,
                                     reliable ? reliable + (frame * geoms + a) * 52 : nullptr,
                                     reliable ? reliable + (frame * geoms + b) * 52 : nullptr);
        }
    }
    fclose(errors);
}
