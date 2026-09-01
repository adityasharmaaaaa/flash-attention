#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "cuda_utils.cuh"

#ifndef TILE
#define TILE 16
#endif

// ---- naive: global memory only, one thread per output element ----
__global__ void naiveMatmul(const float* A, const float* B, float* C,
                             int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

// ---- tiled: shared memory reuse ----
__global__ void tiledMatmul(const float* A, const float* B, float* C,
                             int M, int K, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    int numTiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;

        As[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

float benchmark(void (*launch)(dim3, dim3, const float*, const float*, float*, int, int, int),
                 dim3 grid, dim3 block, const float* A, const float* B, float* C,
                 int M, int K, int N, int iters) {
    // warmup
    launch(grid, block, A, B, C, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        launch(grid, block, A, B, C, M, K, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iters;
}

void launchNaive(dim3 grid, dim3 block, const float* A, const float* B, float* C,
                  int M, int K, int N) {
    naiveMatmul<<<grid, block>>>(A, B, C, M, K, N);
}
void launchTiled(dim3 grid, dim3 block, const float* A, const float* B, float* C,
                  int M, int K, int N) {
    tiledMatmul<<<grid, block>>>(A, B, C, M, K, N);
}

int main(int argc, char** argv) {
    int M = argc > 1 ? atoi(argv[1]) : 1024;
    int K = argc > 2 ? atoi(argv[2]) : 1024;
    int N = argc > 3 ? atoi(argv[3]) : 1024;
    printf("M=%d K=%d N=%d  TILE=%d\n", M, K, N, TILE);

    std::vector<float> A((size_t)M * K), B((size_t)K * N);
    for (auto& x : A) x = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (auto& x : B) x = (float)(rand() % 1000) / 1000.0f - 0.5f;

    float *Ad, *Bd, *Cd_naive, *Cd_tiled;
    CUDA_CHECK(cudaMalloc(&Ad, (size_t)M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Bd, (size_t)K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Cd_naive, (size_t)M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Cd_tiled, (size_t)M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(Ad, A.data(), (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Bd, B.data(), (size_t)K * N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockNaive(16, 16); // fixed baseline config, independent of TILE
    dim3 gridNaive((N + 15) / 16, (M + 15) / 16);
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    float msNaive = benchmark(launchNaive, gridNaive, blockNaive, Ad, Bd, Cd_naive, M, K, N, 10);
    float msTiled = benchmark(launchTiled, grid, block, Ad, Bd, Cd_tiled, M, K, N, 10);

    double flops = 2.0 * M * N * K;
    double gflopsNaive = flops / (msNaive * 1e6);
    double gflopsTiled = flops / (msTiled * 1e6);

    // correctness: compare tiled vs naive (both GPU, fp32, same summation order per-thread)
    std::vector<float> Cn((size_t)M * N), Ct((size_t)M * N);
    CUDA_CHECK(cudaMemcpy(Cn.data(), Cd_naive, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Ct.data(), Cd_tiled, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));
    float maxDiff = 0.0f;
    for (size_t i = 0; i < Cn.size(); i++) maxDiff = fmaxf(maxDiff, fabsf(Cn[i] - Ct[i]));

    printf("naive : %8.3f ms   %8.2f GFLOPS\n", msNaive, gflopsNaive);
    printf("tiled : %8.3f ms   %8.2f GFLOPS\n", msTiled, gflopsTiled);
    printf("speedup: %.2fx\n", msNaive / msTiled);
    printf("max diff (naive vs tiled): %e  [%s]\n", maxDiff, maxDiff < 1e-2f ? "OK" : "CHECK THIS");

    cudaFree(Ad); cudaFree(Bd); cudaFree(Cd_naive); cudaFree(Cd_tiled);
    return 0;
}
