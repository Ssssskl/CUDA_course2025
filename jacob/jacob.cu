#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <cstring>
#ifdef _OPENMP
  #include <omp.h>
#endif

#ifdef USE_FLOAT
typedef float real_t;

__device__ float atomicMaxFloat(float* addr, float val) {
    int* uaddr = (int*)addr;
    int old = *uaddr, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= val) break;
        old = atomicCAS(uaddr, assumed, __float_as_int(val));
    } while (assumed != old);
    return __int_as_float(old);
}
#else
typedef double real_t;

__device__ double atomicMaxDouble(double* addr, double val) {
    unsigned long long* u = (unsigned long long*)addr;
    unsigned long long old = *u, assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) >= val) break;
        old = atomicCAS(u, assumed, __double_as_longlong(val));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

void print_usage(const char* prog){
    printf("Usage: %s [--cpu] [--gpu] [--compare] [-L size] [-i iters]\n"
           "  --cpu      CPU only\n"
           "  --gpu      GPU only (default)\n"
           "  --compare  run both CPU and GPU, compare results\n"
           "  -L size    grid size (default 384)\n"
           "  -i iters   number of iterations (default 20)\n", prog);
}

real_t* alloc_array(int L){
    return (real_t*)malloc((size_t)L*L*L * sizeof(real_t));
}

void init_fields(real_t* A, real_t* B, int L){
    #pragma omp parallel for collapse(3)
    for(int i=0;i<L;i++)for(int j=0;j<L;j++)for(int k=0;k<L;k++){
        size_t idx = (size_t)i*L*L + j*L + k;
        A[idx] = 0;
        if(i==0||j==0||k==0||i==L-1||j==L-1||k==L-1)
            B[idx] = 0;
        else
            B[idx] = (real_t)(4 + i + j + k);
    }
}

__global__ void jacobi_kernel(const real_t* A, real_t* B, int L, real_t* d_eps){
    size_t idx = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    size_t N   = (size_t)L*L*L;
    if(idx>=N) return;
    int i = idx/(L*L), rem = idx%(L*L);
    int j = rem/L, k = rem%L;
    if(i>0&&i<L-1&&j>0&&j<L-1&&k>0&&k<L-1){
        real_t v = (A[idx - L*L] + A[idx - L] + A[idx - 1]
                  + A[idx + 1]   + A[idx + L] + A[idx + L*L])/(real_t)6.0;
        B[idx] = v;
        real_t diff = fabs(v - A[idx]);
        #ifdef USE_FLOAT
            atomicMaxFloat(d_eps, diff);
        #else
            atomicMaxDouble(d_eps, diff);
        #endif
    } else {
        B[idx] = A[idx];
    }
}

__global__ void compute_eps_kernel(const real_t* A, const real_t* B, int L, real_t* d_eps){
    size_t idx = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    size_t N   = (size_t)L*L*L;
    if(idx>=N) return;
    int i = idx/(L*L), rem = idx%(L*L);
    int j = rem/L, k = rem%L;
    if(i>0&&i<L-1&&j>0&&j<L-1&&k>0&&k<L-1){
        real_t diff = fabs(B[idx] - A[idx]);
        #ifdef USE_FLOAT
            atomicMaxFloat(d_eps, diff);
        #else
            atomicMaxDouble(d_eps, diff);
        #endif
    }
}

void cpu_jacobi(real_t* A, real_t* B, int L, int ITER, double &tsec) {
    auto t0 = std::chrono::high_resolution_clock::now();
    size_t N = (size_t)L*L*L;
    real_t *curr = A, *next = B;

    real_t eps = 0;
    #pragma omp parallel for reduction(max:eps)
    for (size_t idx = 0; idx < N; ++idx) {
        int i = idx/(L*L), j = (idx/L)%L, k = idx%L;
        if (i>0&&i<L-1&&j>0&&j<L-1&&k>0&&k<L-1) {
            real_t d = fabs(next[idx] - curr[idx]);
            if (d>eps) eps = d;
        }
    }
    printf(" IT = %4d   EPS = %10.7E\n", 1, eps);

    for (int it = 2; it <= ITER; ++it) {
        std::swap(curr, next);
        #pragma omp parallel for collapse(3)
        for (int i=0; i<L; ++i) for (int j=0; j<L; ++j) for (int k=0; k<L; ++k) {
            size_t idx = (size_t)i*L*L + j*L + k;
            if (i>0&&i<L-1&&j>0&&j<L-1&&k>0&&k<L-1) {
                next[idx] = (curr[idx - L*L] + curr[idx - L] + curr[idx - 1]
                           + curr[idx + 1]   + curr[idx + L] + curr[idx + L*L]) / (real_t)6.0;
            } else {
                next[idx] = curr[idx];
            }
        }
        eps = 0;
        #pragma omp parallel for reduction(max:eps)
        for (size_t idx = 0; idx < N; ++idx) {
            int i = idx/(L*L), j = (idx/L)%L, k = idx%L;
            if (i>0&&i<L-1&&j>0&&j<L-1&&k>0&&k<L-1) {
                real_t d = fabs(next[idx] - curr[idx]);
                if (d>eps) eps = d;
            }
        }
        printf(" IT = %4d   EPS = %10.7E\n", it, eps);
    }

    if (next != B) {
        memcpy(B, next, N * sizeof(real_t));
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    tsec = std::chrono::duration<double>(t1 - t0).count();
}

void gpu_jacobi(real_t* A_h, real_t* B_h, int L, int ITER, double &tsec){
    size_t N    = (size_t)L*L*L;
    size_t Nbyt = N * sizeof(real_t);
    real_t *A_d, *B_d, *d_eps;
    cudaMalloc(&A_d,    Nbyt);
    cudaMalloc(&B_d,    Nbyt);
    cudaMalloc(&d_eps, sizeof(real_t));
    cudaMemcpy(A_d, A_h, Nbyt, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, Nbyt, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (N + threads - 1)/threads;
    real_t eps;

    cudaMemset(d_eps, 0, sizeof(real_t));
    compute_eps_kernel<<<blocks,threads>>>(A_d, B_d, L, d_eps);
    cudaDeviceSynchronize();
    cudaMemcpy(&eps, d_eps, sizeof(real_t), cudaMemcpyDeviceToHost);
    printf(" GPU IT = %4d   EPS = %10.7E\n", 1, eps);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for(int it=2; it<=ITER; ++it){
        std::swap(A_d, B_d);
        cudaMemset(d_eps, 0, sizeof(real_t));
        jacobi_kernel<<<blocks,threads>>>(A_d, B_d, L, d_eps);
        cudaDeviceSynchronize();
        cudaMemcpy(&eps, d_eps, sizeof(real_t), cudaMemcpyDeviceToHost);
        printf(" GPU IT = %4d   EPS = %10.7E\n", it, eps);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop);
    tsec = ms/1000.0f;

    cudaMemcpy(B_h, B_d, Nbyt, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(d_eps);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char** argv){
    int L = 384, ITER = 20;
    bool runCPU=false, runGPU=false, doCmp=false;
    runGPU = true;
    for(int i=1;i<argc;++i){
        if(!strcmp(argv[i],"--cpu"))        { runCPU=true;  runGPU=false; }
        else if(!strcmp(argv[i],"--gpu"))   { runGPU=true;  runCPU=false; }
        else if(!strcmp(argv[i],"--compare")){ doCmp=true; runCPU=runGPU=true; }
        else if(!strcmp(argv[i],"-L") && i+1<argc){ L=atoi(argv[++i]); }
        else if(!strcmp(argv[i],"-i") && i+1<argc){ ITER=atoi(argv[++i]); }
        else { print_usage(argv[0]); return 1; }
    }

    size_t N = (size_t)L*L*L;
    printf("Jacobi3D: %d^3, %d iterations\n", L, ITER);
#ifdef USE_FLOAT
    printf("Data type: float\n");
#else
    printf("Data type: double\n");
#endif
    if(runGPU){
        int dev; cudaGetDevice(&dev);
        cudaDeviceProp p; cudaGetDeviceProperties(&p,dev);
        size_t freeM,totalM; cudaMemGetInfo(&freeM,&totalM);
        printf("GPU: %s, memory %zu MB\n", p.name, totalM/1024/1024);
    }

    real_t *A = alloc_array(L), *B = alloc_array(L);
    init_fields(A, B, L);

    double tCPU=0, tGPU=0;
    if(runCPU){
    #ifdef _OPENMP
        omp_set_num_threads(omp_get_max_threads());
    #endif
        cpu_jacobi(A, B, L, ITER, tCPU);
        printf(" CPU time = %.3fs\n", tCPU);
    }
    if(runGPU){
        real_t *Ag = alloc_array(L), *Bg = alloc_array(L);
        init_fields(Ag, Bg, L);
        gpu_jacobi(Ag, Bg, L, ITER, tGPU);
        printf(" GPU time = %.3fs\n", tGPU);

        if(doCmp){
            real_t maxdiff = 0;
            for(size_t i=0;i<N;++i){
                real_t d = fabs(B[i] - Bg[i]);
                if(d>maxdiff) maxdiff=d;
            }
            printf(" Max diff = %.6e\n", (double)maxdiff);
            printf(" Verification: %s\n", maxdiff<1e-6?"SUCCESSFUL":"UNSUCCESSFUL");
            printf(" Speedup: %.2fx\n", tCPU/tGPU);
        }
        free(Ag); free(Bg);
    }

    free(A); free(B);
    return 0;
}