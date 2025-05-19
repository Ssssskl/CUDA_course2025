#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>
#ifdef _OPENMP
#   include <omp.h>
#endif

#ifdef USE_FLOAT
using real_t = float;
#   define R_EPS 1e-3f
#   define MY_FABS fabsf
#else
using real_t = double;
#   define R_EPS 1e-6
#   define MY_FABS fabs
#endif

#ifdef __CUDACC__
#   define HD __host__ __device__ __forceinline__
#else
#   define HD inline
#endif

HD size_t idx3(int i,int j,int k,int n)
{ return (size_t)i*n*n + (size_t)j*n + k; }

template<typename T> __device__ T atomicMaxReal(T*,T);

template<> __device__ float atomicMaxReal(float* a,float v){
    int* ai=(int*)a; int old=*ai,ass;
    do{ ass=old; if(__int_as_float(ass)>=v) break;
        old=atomicCAS(ai,ass,__float_as_int(v)); }
    while(old!=ass);
    return __int_as_float(old);
}
template<> __device__ double atomicMaxReal(double* a,double v){
    unsigned long long* ai=(unsigned long long*)a,old,ass;
    old=*ai;
    do{ ass=old; if(__longlong_as_double(ass)>=v) break;
        old=atomicCAS(ai,ass,__double_as_longlong(v)); }
    while(old!=ass);
    return __longlong_as_double(old);
}

int L=900, ITER=10;
bool runC=false, runG=true, doCmp=false;

real_t* halloc(){ return (real_t*)malloc((size_t)L*L*L*sizeof(real_t)); }

void init(real_t* a){
#pragma omp parallel for collapse(3)
    for(int i=0;i<L;i++)
        for(int j=0;j<L;j++)
            for(int k=0;k<L;k++){
                size_t p=idx3(i,j,k,L);
                if(i==0||i==L-1||j==0||j==L-1||k==0||k==L-1)
                    a[p]=(real_t)(10.0*i/(L-1)+10.0*j/(L-1)+10.0*k/(L-1));
                else a[p]=0;
            }
}

void cpu_adi(real_t* a,double& sec){
    auto t0=std::chrono::high_resolution_clock::now();
    real_t eps_val=0;
    for(int it=1;it<=ITER;++it){
#pragma omp parallel for collapse(2)
        for(int j=1;j<L-1;j++)
            for(int k=1;k<L-1;k++){
                size_t p=idx3(1,j,k,L);
                for(int i=1;i<L-1;i++,p+=(size_t)L*L)
                    a[p]=(a[p-(size_t)L*L]+a[p+(size_t)L*L])*(real_t)0.5;
            }
#pragma omp parallel for collapse(2)
        for(int i=1;i<L-1;i++)
            for(int k=1;k<L-1;k++){
                size_t p=idx3(i,1,k,L);
                for(int j=1;j<L-1;j++,p+=L)
                    a[p]=(a[p-L]+a[p+L])*(real_t)0.5;
            }
        eps_val=0;
#pragma omp parallel for collapse(2) reduction(max:eps_val)
        for(int i=1;i<L-1;i++)
            for(int j=1;j<L-1;j++){
                size_t p=idx3(i,j,1,L);
                for(int k=1;k<L-1;k++,++p){
                    real_t v=(a[p-1]+a[p+1])*(real_t)0.5;
                    eps_val=fmax(eps_val,MY_FABS(a[p]-v));
                    a[p]=v;
                }
            }
        printf("CPU IT=%3d EPS=%e\n",it,(double)eps_val);
        if(eps_val<R_EPS) break;
    }
    sec=std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count();
}

__global__ void sweep_x(real_t* __restrict__ a, int n){
    int k=blockIdx.x*blockDim.x+threadIdx.x+1;
    int j=blockIdx.y*blockDim.y+threadIdx.y+1;
    if(k>=n-1||j>=n-1) return;
    size_t p=idx3(1,j,k,n);
    for(int i=1;i<n-1;i++,p+=(size_t)n*n)
        a[p]=(a[p-(size_t)n*n]+a[p+(size_t)n*n])*(real_t)0.5;
}
__global__ void sweep_y(real_t* __restrict__ a, int n){
    int k=blockIdx.x*blockDim.x+threadIdx.x+1;
    int i=blockIdx.y*blockDim.y+threadIdx.y+1;
    if(k>=n-1||i>=n-1) return;
    size_t p=idx3(i,1,k,n);
    for(int j=1;j<n-1;j++,p+=n)
        a[p]=(a[p-n]+a[p+n])*(real_t)0.5;
}

__global__ void sweep_z(real_t* __restrict__ a, real_t* __restrict__ g_eps, int n){
    extern __shared__ real_t sh_eps_data[];

    int tj = threadIdx.x;
    int ti = threadIdx.y;
    int j_glob = blockIdx.x * blockDim.x + tj + 1;
    int i_glob = blockIdx.y * blockDim.y + ti + 1;

    int lid = ti * blockDim.x + tj;
    sh_eps_data[lid] = 0.0;

    if (i_glob < n - 1 && j_glob < n - 1) {
        real_t thread_local_eps = 0.0;
        for (int k = 1; k < n - 1; ++k) {
            size_t p_current = idx3(i_glob, j_glob, k, n);
            size_t p_prev_k  = idx3(i_glob, j_glob, k - 1, n);
            size_t p_next_k  = idx3(i_glob, j_glob, k + 1, n);

            real_t v = (a[p_prev_k] + a[p_next_k]) * (real_t)0.5;
            thread_local_eps = fmax(thread_local_eps, MY_FABS(a[p_current] - v));
            a[p_current] = v;
        }
        sh_eps_data[lid] = thread_local_eps;
    }
    __syncthreads();

    int num_threads_in_block = blockDim.x * blockDim.y;
    for (int s = num_threads_in_block / 2; s > 0; s >>= 1) {
        if (lid < s) {
            sh_eps_data[lid] = fmax(sh_eps_data[lid], sh_eps_data[lid + s]);
        }
        __syncthreads();
    }

    if (lid == 0) {
        atomicMaxReal(g_eps, sh_eps_data[0]);
    }
}


void gpu_adi(real_t* h_a,double& sec){
    size_t bytes=(size_t)L*L*L*sizeof(real_t);
    real_t *d_a,*d_eps;
    cudaMalloc(&d_a,bytes);
    cudaMalloc(&d_eps,sizeof(real_t));
    cudaMemcpy(d_a,h_a,bytes,cudaMemcpyHostToDevice);

    dim3 blkXY(32,4);
    dim3 gridXY((L-2+blkXY.x-1)/blkXY.x,(L-2+blkXY.y-1)/blkXY.y);

    dim3 blkZ_corrected(16,16);
    dim3 gridZ_corrected((L-2 + blkZ_corrected.x-1)/blkZ_corrected.x,
                         (L-2 + blkZ_corrected.y-1)/blkZ_corrected.y);
    size_t shBytesZ_corrected = blkZ_corrected.x * blkZ_corrected.y * sizeof(real_t);

    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    real_t h_eps_val = 0.0;
    for(int it=1;it<=ITER;++it){
        cudaMemset(d_eps,0,sizeof(real_t));

        sweep_x<<<gridXY,blkXY>>>(d_a,L);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) { printf("CUDA error after sweep_x: %s\n", cudaGetErrorString(err)); return; }

        sweep_y<<<gridXY,blkXY>>>(d_a,L);
        err = cudaGetLastError();
        if (err != cudaSuccess) { printf("CUDA error after sweep_y: %s\n", cudaGetErrorString(err)); return; }
        
        sweep_z<<<gridZ_corrected, blkZ_corrected, shBytesZ_corrected>>>(d_a,d_eps,L);
        err = cudaGetLastError();
        if (err != cudaSuccess) { printf("CUDA error after sweep_z: %s\n", cudaGetErrorString(err)); return; }

        cudaMemcpy(&h_eps_val,d_eps,sizeof(real_t),cudaMemcpyDeviceToHost);
        printf("GPU IT=%3d EPS=%e\n",it,(double)h_eps_val);
        if(h_eps_val<R_EPS) break;
    }

    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms,t0,t1); sec=ms*1e-3f;

    cudaMemcpy(h_a,d_a,bytes,cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_eps);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

int main(int argc,char** argv){
    for(int i=1;i<argc;++i){
        if(!strcmp(argv[i],"--cpu"))          { runC=true; runG=false; }
        else if(!strcmp(argv[i],"--gpu"))     { runG=true; runC=false; }
        else if(!strcmp(argv[i],"--compare")) { runC=runG=doCmp=true; }
        else if(!strcmp(argv[i],"-L")&&i+1<argc)  L=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-i")&&i+1<argc)  ITER=atoi(argv[++i]);
        else{ printf("Usage: %s [--cpu] [--gpu] [--compare] [-L N] [-i N]\n",argv[0]); return 0; }
    }

    printf("ADI3D: %d^3, %d iter, type=%s\n",L,ITER,sizeof(real_t)==4?"float":"double");
    if(runG){
        cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
        size_t fr,to; cudaMemGetInfo(&fr,&to);
         printf("GPU: %s (Arch: %d.%d, SMs: %d, TotalMem: %.2f MiB, FreeMem: %.2f MiB)\n",
            p.name, p.major, p.minor, p.multiProcessorCount,
            (double)p.totalGlobalMem / (1024 * 1024), (double)fr / (1024*1024) );
    }

    size_t N_elements =(size_t)L*L*L;
    real_t *base_array=halloc(); init(base_array);
    real_t *cA=nullptr,*gA=nullptr; double tC=0,tG=0;

    if(runC){
        cA=halloc();
        memcpy(cA,base_array,N_elements*sizeof(real_t));
        printf("Running CPU version...\n");
        cpu_adi(cA,tC);
        printf("CPU Time: %.3f s\n", tC);
    }
    if(runG){
        gA=halloc();
        memcpy(gA,base_array,N_elements*sizeof(real_t));
        printf("Running GPU version...\n");
        gpu_adi(gA,tG);
        printf("GPU Time: %.3f s\n", tG);
    }

    if(doCmp && runC && runG){
        real_t md=0;
#pragma omp parallel for reduction(max:md)
        for(size_t p=0;p<N_elements;p++) md=fmax(md,MY_FABS(cA[p]-gA[p]));
        printf("Max diff = %e  ==> %s\n",(double)md,(md < (sizeof(real_t) == 4 ? 1e-2 : 1e-5) ? "SUCCESS":"FAIL"));
        if (tG > 0 && tC > 0) {
             printf("Speed-up = %.2fx (CPU %.3f s -> GPU %.3f s)\n",tC/tG,tC,tG);
        }
    }

    free(base_array);
    if(cA) free(cA);
    if(gA) free(gA);
    return 0;
}
