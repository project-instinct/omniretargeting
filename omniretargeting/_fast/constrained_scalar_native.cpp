// Independent stencil QP feasibility polish. All work stays in a scalar band.
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdint>
extern "C" int scalar_project(int64_t n,int64_t m,int64_t width,
 const int64_t* ptr,const int64_t* col,const double* a,const double* b,
 const double* target,double* x,int64_t* iterations,double* remaining) {
 const double rho=100.;
 std::vector<double> h(n*width,0.), z(m), y(m,0.), v(m), rhs(n), previous(m);
 for(int64_t i=0;i<n;i++)h[i*width]=1.;
 for(int64_t r=0;r<m;r++) {
  double value=0.;
  for(int64_t k=ptr[r];k<ptr[r+1];k++) {
   value+=a[k]*target[col[k]];
   for(int64_t q=ptr[r];q<=k;q++) {
    int64_t i=col[k],j=col[q];
    if(i<j || i-j>=width)return -1;
    h[i*width+i-j]+=rho*a[k]*a[q];
   }
  }
  z[r]=std::min(value,b[r]);
 }
 // Row-major lower band Cholesky.
 for(int64_t i=0;i<n;i++)for(int64_t j=std::max<int64_t>(0,i-width+1);j<=i;j++) {
  double value=h[i*width+i-j];
  for(int64_t k=std::max<int64_t>({0,i-width+1,j-width+1});k<j;k++)
   value-=h[i*width+i-k]*h[j*width+j-k];
  if(i==j) {if(!(value>0.))return -2;h[i*width]=std::sqrt(value);}
  else h[i*width+i-j]=value/h[j*width];
 }
 double excess=0.;
 for(int64_t it=0;it<1600;it++) {
  std::copy(target,target+n,rhs.begin());
  for(int64_t r=0;r<m;r++)for(int64_t k=ptr[r];k<ptr[r+1];k++)
   rhs[col[k]]+=a[k]*(rho*z[r]-y[r]);
  for(int64_t i=0;i<n;i++) {
   double value=rhs[i];
   for(int64_t j=std::max<int64_t>(0,i-width+1);j<i;j++)value-=h[i*width+i-j]*x[j];
   x[i]=value/h[i*width];
  }
  for(int64_t i=n;i-->0;) {
   double value=x[i];
   for(int64_t j=i+1;j<std::min(n,i+width);j++)value-=h[j*width+j-i]*x[j];
   x[i]=value/h[i*width];
  }
  double primal=0.,dual=0.;excess=0.;
  for(int64_t r=0;r<m;r++) {
   double value=0.;for(int64_t k=ptr[r];k<ptr[r+1];k++)value+=a[k]*x[col[k]];
   double old=z[r],over=1.6*value-.6*old;
   z[r]=std::min(over+y[r]/rho,b[r]);y[r]+=rho*(over-z[r]);
   primal=std::max(primal,std::abs(value-z[r]));
   dual=std::max(dual,std::abs(rho*(z[r]-old)));
   excess=std::max(excess,value-b[r]);
  }
  *iterations=it+1;
  if(primal<1e-10 && dual<1e-9)break;
 }
 *remaining=excess;return 0;
}
