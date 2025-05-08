#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>
#ifdef _OPENMP
  #include <omp.h>
#endif

#ifdef USE_FLOAT
  typedef float real_t;
  #define REAL_MAXEPS 1e-3f
#else
  typedef double real_t;
  #define REAL_MAXEPS 1e-6
#endif

int L        = 900;
int ITER     = 20;
bool runCPU  = false;
bool runGPU  = false;
bool doCmp   = false;

void print_usage(const char* prog) {
    printf("Usage: %s [--cpu] [--gpu] [--compare] [-L size] [-i iters]\n"
           "  --cpu      CPU only\n"
           "  --gpu      GPU only (default)\n"
           "  --compare  run both CPU and GPU, compare results\n"
           "  -L size    grid size (default 900)\n"
           "  -i iters   number of iterations (default 20)\n", prog);
}

real_t* alloc_array() {
    return (real_t*)malloc(sizeof(real_t)*(size_t)L*L*L);
}

void init_fields(real_t* a) {
    #pragma omp parallel for collapse(3)
    for(int i = 0; i < L; ++i)
    for(int j = 0; j < L; ++j)
    for(int k = 0; k < L; ++k) {
        size_t idx = (size_t)i*L*L + j*L + k;
        if (i==0||i==L-1||j==0||j==L-1||k==0||k==L-1)
            a[idx] = (real_t)(10.0*i/(L-1) + 10.0*j/(L-1) + 10.0*k/(L-1));
        else
            a[idx] = 0;
    }
}

void cpu_adi(real_t* a, double &tsec) {
    real_t *b = alloc_array();
    auto t0 = std::chrono::high_resolution_clock::now();
    real_t eps = 0;

    for(int it = 1; it <= ITER; ++it) {
        // X-sweep
        #pragma omp parallel for collapse(3)
        for(int i = 1; i < L-1; ++i)
        for(int j = 1; j < L-1; ++j)
        for(int k = 1; k < L-1; ++k) {
            size_t idx = (size_t)i*L*L + j*L + k;
            b[idx] = (a[idx - L*L] + a[idx + L*L]) * (real_t)0.5;
        }
        // swap a and b
        {
            real_t* tmp = a;
            a = b;
            b = tmp;
        }

        // Y-sweep
        #pragma omp parallel for collapse(3)
        for(int i = 1; i < L-1; ++i)
        for(int j = 1; j < L-1; ++j)
        for(int k = 1; k < L-1; ++k) {
            size_t idx = (size_t)i*L*L + j*L + k;
            b[idx] = (a[idx - L] + a[idx + L]) * (real_t)0.5;
        }
        // swap a and b
        {
            real_t* tmp = a;
            a = b;
            b = tmp;
        }

        // Z-sweep + compute eps
        eps = 0;
        #pragma omp parallel for collapse(3) reduction(max:eps)
        for(int i = 1; i < L-1; ++i)
        for(int j = 1; j < L-1; ++j)
        for(int k = 1; k < L-1; ++k) {
            size_t idx = (size_t)i*L*L + j*L + k;
            real_t v = (a[idx - 1] + a[idx + 1]) * (real_t)0.5;
            b[idx] = v;
            real_t d = fabs(a[idx] - v);
            if (d > eps) eps = d;
        }
        // swap a and b
        {
            real_t* tmp = a;
            a = b;
            b = tmp;
        }

        printf("CPU IT=%3d EPS=%e\n", it, eps);
        if (eps < REAL_MAXEPS) break;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    tsec = std::chrono::duration<double>(t1 - t0).count();
    free(b);
}

__global__ void adi_x_sweep(const real_t* a, real_t* b, int L) {
    int k = blockIdx.x*blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y*blockDim.y + threadIdx.y + 1;
    int i = blockIdx.z*blockDim.z + threadIdx.z + 1;
    if (i < L-1 && j < L-1 && k < L-1) {
        size_t idx = (size_t)i*L*L + j*L + k;
        b[idx] = (a[idx - L*L] + a[idx + L*L]) * (real_t)0.5;
    }
}

__global__ void adi_y_sweep(const real_t* a, real_t* b, int L) {
    int k = blockIdx.x*blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y*blockDim.y + threadIdx.y + 1;
    int i = blockIdx.z*blockDim.z + threadIdx.z + 1;
    if (i < L-1 && j < L-1 && k < L-1) {
        size_t idx = (size_t)i*L*L + j*L + k;
        b[idx] = (a[idx - L] + a[idx + L]) * (real_t)0.5;
    }
}

__global__ void adi_z_sweep(const real_t* a, real_t* b, real_t* block_eps, int L) {
    extern __shared__ real_t sdata[];
    int tid       = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    int blockSize = blockDim.x*blockDim.y*blockDim.z;

    int k = blockIdx.x*blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y*blockDim.y + threadIdx.y + 1;
    int i = blockIdx.z*blockDim.z + threadIdx.z + 1;
    real_t leps = 0;

    if (i < L-1 && j < L-1 && k < L-1) {
        size_t idx = (size_t)i*L*L + j*L + k;
        real_t v = (a[idx - 1] + a[idx + 1]) * (real_t)0.5;
        b[idx] = v;
        leps = fabs(a[idx] - v);
    }

    sdata[tid] = leps;
    __syncthreads();
    for(int s = blockSize/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        int bid = blockIdx.x + gridDim.x*(blockIdx.y + gridDim.y*blockIdx.z);
        block_eps[bid] = sdata[0];
    }
}

void gpu_adi(real_t* a_h, double &tsec) {
    size_t bytes = sizeof(real_t)*(size_t)L*L*L;
    real_t *d_a, *d_b;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMemcpy(d_a, a_h, bytes, cudaMemcpyHostToDevice);

    dim3 blk(8,8,8);
    dim3 grid((L-2+blk.x-1)/blk.x, (L-2+blk.y-1)/blk.y, (L-2+blk.z-1)/blk.z);
    int numBlocks = grid.x * grid.y * grid.z;

    real_t *d_eps;
    cudaMalloc(&d_eps, numBlocks * sizeof(real_t));
    real_t *h_eps = (real_t*)malloc(numBlocks * sizeof(real_t));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    real_t eps = 0;
    for(int it = 1; it <= ITER; ++it) {
        adi_x_sweep<<<grid,blk>>>(d_a, d_b, L);
        cudaDeviceSynchronize();
        adi_y_sweep<<<grid,blk>>>(d_b, d_a, L);
        cudaDeviceSynchronize();
        size_t shmem = blk.x*blk.y*blk.z * sizeof(real_t);
        adi_z_sweep<<<grid,blk,shmem>>>(d_a, d_b, d_eps, L);
        cudaDeviceSynchronize();
        // swap device buffers
        {
            real_t* tmp = d_a;
            d_a = d_b;
            d_b = tmp;
        }

        cudaMemcpy(h_eps, d_eps, numBlocks*sizeof(real_t), cudaMemcpyDeviceToHost);
        eps = 0;
        for(int i = 0; i < numBlocks; ++i) {
            if (h_eps[i] > eps) eps = h_eps[i];
        }
        printf("GPU IT=%3d EPS=%e\n", it, eps);
        if (eps < REAL_MAXEPS) break;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop);
    tsec = ms/1000.0f;

    cudaMemcpy(a_h, d_a, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_eps);
    free(h_eps);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
    runGPU = true;
    for(int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--cpu"))     { runCPU = true; runGPU = false; }
        else if (!strcmp(argv[i], "--gpu"))     { runGPU = true; runCPU = false; }
        else if (!strcmp(argv[i], "--compare")) { doCmp  = true; runCPU = runGPU = true; }
        else if (!strcmp(argv[i], "-L") && i+1<argc) { L    = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "-i") && i+1<argc) { ITER = atoi(argv[++i]); }
        else { print_usage(argv[0]); return 1; }
    }

    size_t N = (size_t)L*L*L;
    printf("ADI3D: %d^3, %d iterations\n", L, ITER);
  #ifdef USE_FLOAT
    printf("Data type: float\n");
  #else
    printf("Data type: double\n");
  #endif

    if (runGPU) {
        int dev; cudaGetDevice(&dev);
        cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
        size_t freeMem, totalMem;
        cudaMemGetInfo(&freeMem, &totalMem);
        printf("GPU: %s, free memory %zu MiB / %zu MiB\n",
               prop.name, freeMem>>20, totalMem>>20);
    }

    real_t *h_base = alloc_array();
    init_fields(h_base);

    real_t *h_cpu = nullptr, *h_gpu = nullptr;
    double tCPU = 0, tGPU = 0;

    if (runCPU) {
        h_cpu = alloc_array();
        memcpy(h_cpu, h_base, N*sizeof(real_t));
        cpu_adi(h_cpu, tCPU);
        printf("CPU time = %.3fs\n", tCPU);
    }
    if (runGPU) {
        h_gpu = alloc_array();
        memcpy(h_gpu, h_base, N*sizeof(real_t));
        gpu_adi(h_gpu, tGPU);
        printf("GPU time = %.3fs\n", tGPU);
    }

    if (doCmp && runCPU && runGPU) {
        real_t maxdiff = 0;
        for(size_t i = 0; i < N; ++i) {
            real_t d = fabs(h_cpu[i] - h_gpu[i]);
            if (d > maxdiff) maxdiff = d;
        }
        printf("Max diff = %.6e\n", (double)maxdiff);
        printf("Verification: %s\n", maxdiff < (real_t)1e-6 ? "SUCCESSFUL" : "UNSUCCESSFUL");
        printf("Speedup: %.2fx\n", tCPU / tGPU);
    }

    free(h_base);
    if (h_cpu) free(h_cpu);
    if (h_gpu) free(h_gpu);
    return 0;
}