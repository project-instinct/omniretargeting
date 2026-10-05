// Window-only exact support and shell pruning. Shared reference arithmetic.
#include "bulk_geometry_native.cpp"

static double support_bound(const double* d, const double* box) {
  return (d[0]*box[d[0]>=0 ? 3 : 0] + d[1]*box[d[1]>=0 ? 4 : 1])
         + d[2]*box[d[2]>=0 ? 5 : 2];
}

extern "C" void window_support(int64_t count, const double* directions,
                               const double* points, const int64_t* order,
                               const double* boxes, const int64_t* nodes,
                               double slack, int64_t* choices, double* margins) {
  for (int64_t i=0; i<count; ++i) {
    const double* d=directions+3*i;
    double best=-INFINITY, second=-INFINITY;
    int64_t choice=0, stack[128], size=1;
    stack[0]=0;
    // The balanced tree has depth <= 63 for an int64 vertex count.
    while (size) {
      int64_t node=stack[--size];
      if (support_bound(d, boxes+6*node)+slack < second) continue;
      const int64_t* info=nodes+4*node;
      if (info[2]<0) {
        for (int64_t k=info[0]; k<info[1]; ++k) {
          int64_t j=order[k];
          const double* p=points+3*j;
          double v=(d[0]*p[0]+d[1]*p[1])+d[2]*p[2];
          if (v>best) { second=best; best=v; choice=j; }
          else {
            if (v>second) second=v;
            if (v==best && j<choice) choice=j;
          }
        }
      } else {
        int64_t a=info[2], b=info[3];
        if (support_bound(d,boxes+6*a)>support_bound(d,boxes+6*b)) {
          stack[size++]=b; stack[size++]=a;
        } else { stack[size++]=a; stack[size++]=b; }
      }
    }
    choices[i]=choice; margins[i]=best-second;
  }
}

static float bounded_lookup(const float* grid, const int64_t* shape,
                            const double* lower, double spacing,
                            const double* point, const float* bounds,
                            const int64_t* coarse, int64_t block,
                            double best, double scale, bool later, int64_t* counters) {
  int64_t base[3]; double t[3];
  for (int a=0; a<3; ++a) {
    double x=(point[a]-lower[a])/spacing;
    if (!(x>=0 && x<=shape[a]-1)) return NAN;
    double f=std::floor(x);
    base[a]=static_cast<int64_t>(f); t[a]=x-f;
  }
  int64_t cell[3];
  // Avoid three runtime integer divisions at every sample in the hot path.
  for (int a=0; a<3; ++a)
    cell[a]=block==2 ? base[a]>>1 : block==4 ? base[a]>>2 : base[a]/block;
  int64_t index=(cell[0]*coarse[1]+cell[1])*coarse[2]+cell[2];
  // Float32 bounds round downward beyond the double interpolation error.
  // Positive scale preserves ordering; other scales use the full reference.
  double bound=static_cast<double>(bounds[index])*scale;
  if (scale>0 && (bound>best || (later && bound>=best))) {
    ++counters[0];
    return NAN;
  }
  ++counters[1];
  double total=0;
  for (int c=0; c<8; ++c) {
    int dx=(c>>2)&1, dy=(c>>1)&1, dz=c&1;
    int64_t x=std::min(base[0]+dx,shape[0]-1);
    int64_t y=std::min(base[1]+dy,shape[1]-1);
    int64_t z=std::min(base[2]+dz,shape[2]-1);
    double v=grid[(x*shape[1]+y)*shape[2]+z];
    v*=dx ? t[0] : 1-t[0];
    v*=dy ? t[1] : 1-t[1];
    v*=dz ? t[2] : 1-t[2];
    total+=v;
  }
  return static_cast<float>(total);
}

extern "C" void window_shell(int64_t count, int64_t samples, int64_t geom,
                             int64_t ngeom, const int64_t* frames,
                             const double* local, const double* position,
                             const double* rotation, const double* field_rotation,
                             const double* translation, const double* scales,
                             const float* grid, const int64_t* shape,
                             const double* lower, double spacing,
                             const float* bounds, const int64_t* coarse,
                             int64_t block, const int64_t* seeds,
                             double* deepest, int64_t* choices, int64_t* counters) {
  for (int64_t i=0; i<count; ++i) {
    int64_t f=frames[i];
    const double* p=position+(f*ngeom+geom)*3;
    const double* r=rotation+(f*ngeom+geom)*9;
    double best=INFINITY, scale=scales ? scales[f] : 1;
    int64_t choice=0;
    auto sample=[&](int64_t j, bool bounded) {
      const double* v=local+3*j;
      double world[3], field[3];
      for (int a=0; a<3; ++a)
        world[a]=p[a]+((r[3*a]*v[0]+r[3*a+1]*v[1])+r[3*a+2]*v[2]);
      if (field_rotation) {
        const double* fr=field_rotation+9*f;
        double u[3];
        for (int a=0; a<3; ++a) u[a]=world[a]-translation[3*f+a];
        for (int a=0; a<3; ++a)
          field[a]=((fr[a]*u[0]+fr[3+a]*u[1])+fr[6+a]*u[2])/scale;
      } else std::copy(world,world+3,field);
      return static_cast<double>(bounded ? bounded_lookup(grid,shape,lower,spacing,
          field,bounds,coarse,block,best,scale,j>choice,counters) : lookup(grid,shape,lower,spacing,field))*scale;
    };
    int64_t seed=seeds[i];
    if (seed>=0 && seed<samples) {
      double value=sample(seed,false);
      ++counters[1];
      if (std::isfinite(value)) { best=value; choice=seed; }
    }
    for (int64_t j=0; j<samples; ++j) {
      // Earlier equal samples must still be queried to restore first argmin.
      double value=sample(j,true);
      if (std::isfinite(value) && (value<best || (value==best && j<choice))) {
        best=value; choice=j;
      }
    }
    deepest[i]=best; choices[i]=choice;
  }
}
