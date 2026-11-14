// FP32 resident transformer kernels. No quadratic attention workspace.
extern "C" __global__ void binary(const float* a,const float* b,float* y,int n,int kind,float scale){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=kind==0?a[i]+b[i]:kind==1?a[i]*b[i]:a[i]*scale;
}
extern "C" __global__ void transpose(const float* a,float* y,int r,int c){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<r*c)y[(i%c)*r+i/c]=a[i];}
extern "C" __global__ void activation(const float* u,const float* v,const float* g,float* y,float* dv,int n,int kind,int backward){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;float x=u[i],s=1.f/(1.f+expf(-x));
 if(kind==1){y[i]=backward?g[i]*v[i]*(s+x*s*(1-s)):x*s*v[i];if(backward)dv[i]=g[i]*x*s;}
 else {float t=tanhf(.7978845608028654f*(x+.044715f*x*x*x));y[i]=backward?g[i]*(.5f*(1+t)+.5f*x*(1-t*t)*.7978845608028654f*(1+3*.044715f*x*x)):.5f*x*(1+t);}
}
extern "C" __global__ void norm_forward(const float* x,const float* w,const float* b,float* norm,float* inv,float* y,int h,int n,float eps,int center){
 int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=n)return;float mean=0,ss=0;if(center){for(int d=0;d<h;d++)mean+=x[d*n+t];mean/=h;}for(int d=0;d<h;d++){float a=x[d*n+t]-mean;ss+=a*a;}float r=rsqrtf(ss/h+eps);inv[t]=r;for(int d=0;d<h;d++){float a=(x[d*n+t]-mean)*r;norm[d*n+t]=a;y[d*n+t]=a*w[d]+(center?b[d]:0);}
}
extern "C" __global__ void norm_backward(const float* g,const float* norm,const float* inv,const float* w,float* dx,int h,int n,int center){
 int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=n)return;float dot=0,sum=0;for(int d=0;d<h;d++){float a=g[d*n+t]*w[d];dot+=a*norm[d*n+t];sum+=a;}for(int d=0;d<h;d++)dx[d*n+t]=inv[t]*(g[d*n+t]*w[d]-norm[d*n+t]*dot/h-(center?sum/h:0));
}
extern "C" __global__ void norm_parameters(const float* g,const float* norm,float* dw,float* db,int h,int n,int center){int d=blockIdx.x*blockDim.x+threadIdx.x;if(d>=h)return;float a=0,b=0;for(int t=0;t<n;t++){a+=g[d*n+t]*norm[d*n+t];b+=g[d*n+t];}dw[d]+=a;if(center)db[d]+=b;}
extern "C" __global__ void rope(float* x,int h,int dim,int n,int offset,int inverse){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=h*dim/2*n)return;int t=i%n,p=i/n,d=(p%(dim/2))*2,head=p/(dim/2);float angle=(offset+t)*powf(10000.f,-(float)d/dim)*(inverse?-1:1),s,c;sincosf(angle,&s,&c);int a=(head*dim+d)*n+t,b=a+n;float u=x[a],v=x[b];x[a]=u*c-v*s;x[b]=u*s+v*c;}
// Each output element performs online softmax: O(1) workspace and stable max rescaling.
// One cooperative block per query/head; online softmax and linear shared workspace.
extern "C" __global__ void attention(const float* q,const float* k,const float* v,float* y,int heads,int kv,int dim,int n,int keys,int offset){
 int t=blockIdx.x%n,h=blockIdx.x/n,kh=h/(heads/kv),tid=threadIdx.x;
 extern __shared__ float memory[];float* reduction=memory;float* accum=memory+256;
 __shared__ float state[4];
 if(tid==0){state[0]=-3.402823466e+38F;state[1]=0;}
 for(int d=tid;d<dim;d+=256)accum[d]=0;__syncthreads();
 for(int s=0;s<=offset+t;s++){
  float z=0;for(int d=tid;d<dim;d+=256)z+=q[(h*dim+d)*n+t]*k[(kh*dim+d)*keys+s];reduction[tid]=z;__syncthreads();
  for(int step=128;step>0;step/=2){if(tid<step)reduction[tid]+=reduction[tid+step];__syncthreads();}
  if(tid==0){z=reduction[0]*rsqrtf((float)dim);float next=fmaxf(state[0],z);state[2]=expf(state[0]-next);state[3]=expf(z-next);state[1]=state[1]*state[2]+state[3];state[0]=next;}__syncthreads();
  for(int d=tid;d<dim;d+=256)accum[d]=accum[d]*state[2]+state[3]*v[(kh*dim+d)*keys+s];__syncthreads();
 }
 for(int d=tid;d<dim;d+=256)y[(h*dim+d)*n+t]=accum[d]/state[1];
}
extern "C" __global__ void attention_backward(const float* q,const float* k,const float* v,const float* y,const float* g,float* dq,float* dk,float* dv,int heads,int kv,int dim,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=heads*n)return;int t=i%n,h=i/n,kh=h/(heads/kv);float max=-3.402823466e+38F,denom=0,scale=rsqrtf((float)dim),delta=0;
 for(int d=0;d<dim;d++)delta+=g[(h*dim+d)*n+t]*y[(h*dim+d)*n+t];
 for(int s=0;s<=t;s++){float z=0;for(int d=0;d<dim;d++)z+=q[(h*dim+d)*n+t]*k[(kh*dim+d)*n+s];max=fmaxf(max,z*scale);}
 for(int s=0;s<=t;s++){float z=0;for(int d=0;d<dim;d++)z+=q[(h*dim+d)*n+t]*k[(kh*dim+d)*n+s];denom+=expf(z*scale-max);}
 for(int s=0;s<=t;s++){float z=0,dp=0;for(int d=0;d<dim;d++){z+=q[(h*dim+d)*n+t]*k[(kh*dim+d)*n+s];dp+=g[(h*dim+d)*n+t]*v[(kh*dim+d)*n+s];}float p=expf(z*scale-max)/denom,ds=p*(dp-delta)*scale;for(int d=0;d<dim;d++){dq[(h*dim+d)*n+t]+=ds*k[(kh*dim+d)*n+s];atomicAdd(&dk[(kh*dim+d)*n+s],ds*q[(h*dim+d)*n+t]);atomicAdd(&dv[(kh*dim+d)*n+s],p*g[(h*dim+d)*n+t]);}}
}
extern "C" __global__ void embedding(const float* ids,const float* w,float* y,int h,int vocab,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<h*n)y[i]=w[(i/n)*vocab+(int)ids[i%n]];}
extern "C" __global__ void embedding_backward(const float* ids,const float* g,float* dw,int h,int vocab,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<h*n)atomicAdd(&dw[(i/n)*vocab+(int)ids[i%n]],g[i]);}
extern "C" __global__ void slice(const float* x,float* y,int rows,int cols,int offset,int width){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<rows*width)y[i]=x[(i/width)*cols+offset+i%width];}
extern "C" __global__ void append(const float* x,float* y,int rows,int cols,int offset,int width,int add){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<rows*width){int j=(i/width)*cols+offset+i%width;if(add)y[j]+=x[i];else y[j]=x[i];}}
extern "C" __global__ void xent(const float* logits,const float* labels,float* delta,float* losses,int classes,int n){int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=n)return;float max=-3.402823466e+38F,sum=0;for(int c=0;c<classes;c++)max=fmaxf(max,logits[c*n+t]);for(int c=0;c<classes;c++)sum+=expf(logits[c*n+t]-max);losses[t]=(logf(sum)+max-logits[(int)labels[t]*n+t])/n;for(int c=0;c<classes;c++)delta[c*n+t]=(expf(logits[c*n+t]-max)/sum-(c==(int)labels[t]))/n;}
extern "C" __global__ void squared_sum(const float* x,float* y,int n){__shared__ float buf[256];int tid=threadIdx.x;float sum=0;for(int i=tid;i<n;i+=256)sum+=x[i]*x[i];buf[tid]=sum;__syncthreads();for(int d=128;d>0;d/=2){if(tid<d)buf[tid]+=buf[tid+d];__syncthreads();}if(tid==0)y[0]=buf[0];}
extern "C" __global__ void route(const float* logits,float* prob,float* score,float* weights,float* freq,int e,int n,int topk,int sigmoid){int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=n)return;float max=-3.402823466e+38F,sum=0;for(int j=0;j<e;j++)max=fmaxf(max,logits[j*n+t]);for(int j=0;j<e;j++)sum+=expf(logits[j*n+t]-max);for(int j=0;j<e;j++){prob[j*n+t]=expf(logits[j*n+t]-max)/sum;score[j*n+t]=sigmoid?1.f/(1.f+expf(-logits[j*n+t])):prob[j*n+t];weights[j*n+t]=0;}float denom=0;for(int k=0;k<topk;k++){int best=-1;for(int j=0;j<e;j++)if(weights[j*n+t]==0&&(best<0||score[j*n+t]>score[best*n+t]))best=j;weights[best*n+t]=-1;denom+=score[best*n+t];atomicAdd(&freq[best],1.f/(n*topk));}for(int j=0;j<e;j++)if(weights[j*n+t]<0)weights[j*n+t]=denom>0?score[j*n+t]/denom:1.f/topk;}
extern "C" __global__ void dispatch(const float* weights,float* slots,float* counts,int e,int n,int cap){int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=e)return;int count=0;for(int t=0;t<n;t++)if(weights[j*n+t]>0&&count<cap)slots[j*cap+count++]=t;counts[j]=count;}
extern "C" __global__ void gather(const float* x,const float* slots,float* batch,int h,int n,int cap,int expert,int count){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<h*count)batch[i]=x[(i/count)*n+(int)slots[expert*cap+i%count]];}
extern "C" __global__ void combine(const float* y,const float* weights,const float* slots,float* out,int h,int n,int cap,int expert,int count){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<h*count){int t=(int)slots[expert*cap+i%count];out[(i/count)*n+t]+=y[i]*weights[expert*n+t];}}
extern "C" __global__ void expert_delta(const float* g,const float* y,const float* weights,const float* slots,float* dy,float* dw,int h,int n,int cap,int expert,int count){int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=count)return;int t=(int)slots[expert*cap+b];float sum=0;for(int d=0;d<h;d++){float v=g[d*n+t];dy[d*count+b]=v*weights[expert*n+t];sum+=v*y[d*count+b];}dw[expert*n+t]=sum;}
extern "C" __global__ void scatter_gradient(const float* batch,const float* slots,float* dx,int h,int n,int cap,int expert,int count){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<h*count)dx[(i/count)*n+(int)slots[expert*cap+i%count]]+=batch[i];}
extern "C" __global__ void route_backward(const float* dw,const float* weights,const float* prob,const float* score,const float* freq,float* dl,int e,int n,int sigmoid,float balance){int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=n)return;float dot=0,mean=0;for(int j=0;j<e;j++){dot+=weights[j*n+t]*dw[j*n+t];mean+=prob[j*n+t]*freq[j];}for(int j=0;j<e;j++)dl[j*n+t]=weights[j*n+t]*(dw[j*n+t]-dot)*(sigmoid?1-score[j*n+t]:1)+balance*e/n*prob[j*n+t]*(freq[j]-mean);}
extern "C" __global__ void balance_loss(const float* prob,const float* freq,float* loss,int e,int n,float balance){int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=n)return;float sum=0;for(int j=0;j<e;j++)sum+=prob[j*n+t]*freq[j];loss[t]=balance*e/n*sum;}
