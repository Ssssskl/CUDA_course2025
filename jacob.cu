#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#ifdef _OPENMP
  #include <omp.h>
#endif

#ifdef USE_FLOAT
  typedef float real_t;
#else
  typedef double real_t;
#endif

int L       = 384;
int ITER    = 20;
bool runCPU = false;
bool runGPU = false;
bool doCmp  = false;

void print_usage(const char* prog) {
    printf("Usage: %s [--cpu] [--gpu] [--compare] [-L size] [-i iters]\n"
           "  --cpu      CPU only\n"
           "  --gpu      GPU only (default)\n"
           "  --compare  run both CPU and GPU, compare results\n"
           "  -L size    grid size (default 384)\n"
           "  -i iters   number of iterations (default 20)\n", prog);
}

real_t* alloc_array() {
    return (real_t*)malloc(sizeof(real_t)*(size_t)L*L*L);
}
void init_fields(real_t* A, real_t* B) {
    for(int i=0; i<L; i++)
    for(int j=0; j<L; j++)
    for(int k=0; k<L; k++) {
        size_t idx = (size_t)i*L*L + j*L + k;
        A[idx] = 0;
        if (i==0||j==0||k==0||i==L-1||j==L-1||k==L-1)
            B[idx] = 0;
        else
            B[idx] = (real_t)(4 + i + j + k);
    }
}

void cpu_jacobi(real_t* A, real_t* B, double &tsec) {
    auto t0 = std::chrono::high_resolution_clock::now();
    real_t eps, tol = (real_t)0.5;
    for(int it=1; it<=ITER; ++it) {
        eps = 0;
        #pragma omp parallel for reduction(max:eps)
        for(size_t idx=0; idx<(size_t)L*L*L; ++idx) {
            int i = idx/(L*L), j = (idx/L)%L, k = idx%L;
            if(i>0&&i<L-1&&j>0&&j<L-1&&k>0&&k<L-1) {
                real_t d = fabs(B[idx] - A[idx]);
                if (d>eps) eps = d;
            }
        }
        real_t* tmp = A; A = B; B = tmp;
        #pragma omp parallel for
        for (size_t idx = 0; idx < (size_t)(L-2)*(L-2)*(L-2); ++idx) {
            int i = idx / ((L-2)*(L-2)) + 1;
            int j = (idx / (L-2)) % (L-2) + 1;
            int k = idx % (L-2) + 1;
            size_t index = (size_t)i*L*L + j*L + k;
            B[index] = (A[index - L*L] + A[index - L] + A[index - 1] +
                        A[index + 1] + A[index + L] + A[index + L*L]) / 6.0;
        }
        if (eps < tol) break;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    tsec = std::chrono::duration<double>(t1 - t0).count();
}

__global__ void jacobi_kernel(const real_t* A, real_t* B, int L) {
    size_t idx = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    size_t N = (size_t)L*L*L;
    if (idx >= N) return;
    int i = idx/(L*L), rem = idx%(L*L);
    int j = rem/L, k = rem%L;
    if(i>0&&i<L-1&&j>0&&j<L-1&&k>0&&k<L-1) {
        B[idx] = (A[idx - L*L] + A[idx - L] + A[idx - 1]
                + A[idx + 1]   + A[idx + L] + A[idx + L*L])
               / (real_t)6.0;
    } else {
        B[idx] = A[idx];
    }
}

void gpu_jacobi(real_t* A_h, real_t* B_h, double &tsec) {
    size_t Nbytes = sizeof(real_t)*(size_t)L*L*L;
    real_t *A_d, *B_d;
    cudaMalloc(&A_d, Nbytes);
    cudaMalloc(&B_d, Nbytes);
    cudaMemcpy(A_d, A_h, Nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, Nbytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = ((size_t)L*L*L + threads - 1)/threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int it=0; it<ITER; ++it) {
        real_t* tmp = A_d; A_d = B_d; B_d = tmp;
        jacobi_kernel<<<blocks,threads>>>(A_d, B_d, L);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop);
    tsec = ms/1000.0f;

    cudaMemcpy(B_h, B_d, Nbytes, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
    runGPU = true;
    for(int i=1; i<argc; ++i) {
        if (!strcmp(argv[i],"--cpu"))       { runCPU=true;  runGPU=false; }
        else if (!strcmp(argv[i],"--gpu"))  { runGPU=true;  runCPU=false; }
        else if (!strcmp(argv[i],"--compare")) { doCmp=true; runCPU=runGPU=true; }
        else if (!strcmp(argv[i],"-L") && i+1<argc) { L = atoi(argv[++i]); }
        else if (!strcmp(argv[i],"-i") && i+1<argc) { ITER = atoi(argv[++i]); }
        else { print_usage(argv[0]); return 1; }
    }

    size_t N = (size_t)L*L*L;
    printf("Jacobi3D: %d^3, %d iterations\n", L, ITER);
  #ifdef USE_FLOAT
    printf("Data type: float\n");
  #else
    printf("Data type: double\n");
  #endif

    if (runGPU) {
        int dev; cudaGetDevice(&dev);
        cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
        printf("GPU: %s, memory %zu MB\n",
               p.name, cudaMemGetInfo?
               [](){ size_t f,t; cudaMemGetInfo(&f,&t); return t/1024/1024; }()
               :0);
    }

    real_t *A = alloc_array(), *B = alloc_array();
    init_fields(A,B);

    double tCPU=0, tGPU=0;

    if (runCPU) {
    #ifdef _OPENMP
        omp_set_num_threads(omp_get_max_threads());
    #endif
        cpu_jacobi(A,B,tCPU);
        printf("CPU time = %.3fs\n", tCPU);
    }

    if (runGPU) {
        real_t *Ag = alloc_array(), *Bg = alloc_array();
        init_fields(Ag,Bg);
        gpu_jacobi(Ag,Bg,tGPU);
        printf("GPU time = %.3fs\n", tGPU);

        if (doCmp) {
            real_t maxdiff = 0;
            for(size_t i=0; i<N; ++i) {
                real_t d = fabs(B[i] - Bg[i]);
                if (d>maxdiff) maxdiff=d;
            }
            printf("Max diff = %.6e\n", (double)maxdiff);
            printf("Verification: %s\n", maxdiff < 1e-6 ? "SUCCESSFUL" : "UNSUCCESSFUL");
            printf("Speedup: %.2fx\n", tCPU / tGPU);
        }
        free(Ag); free(Bg);
    }

    free(A); free(B);
    return 0;
}