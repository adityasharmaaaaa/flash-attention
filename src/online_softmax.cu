#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <vector>
#include <limits>
#include "cuda_utils.cuh"

#ifndef ROW_TILE
#define ROW_TILE 8   // width of each "tile" the online algorithm consumes at a time
#endif

__global__ void onlineSoftmaxRows(const float* S, float* P, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;

    float m = -INFINITY;
    float l = 0.0f;

    for (int tileStart = 0; tileStart < cols; tileStart += ROW_TILE) {
        int tileEnd = min(tileStart + ROW_TILE, cols);

        // local max of this tile
        float mTile = -INFINITY;
        for (int j = tileStart; j < tileEnd; j++) {
            mTile = fmaxf(mTile, S[i * cols + j]);
        }

        float mNew = fmaxf(m, mTile);
        float correction = expf(m - mNew);   // ==0 cleanly when m==-INFINITY (first tile)

        float tileSum = 0.0f;
        for (int j = tileStart; j < tileEnd; j++) {
            tileSum += expf(S[i * cols + j] - mNew);
        }

        l = l * correction + tileSum;
        m = mNew;
    }

    // second pass: now m, l are the TRUE row max/normalizer
    for (int j = 0; j < cols; j++) {
        P[i * cols + j] = expf(S[i * cols + j] - m) / l;
    }
}

static std::vector<float> readBin(const char* path, size_t count) {
    std::vector<float> data(count);
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "Failed to open %s\n", path); exit(1); }
    f.read(reinterpret_cast<char*>(data.data()), count * sizeof(float));
    return data;
}
static void writeBin(const char* path, const std::vector<float>& data) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
}

int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s rows cols s.bin p_out.bin\n", argv[0]);
        return 1;
    }
    int rows = atoi(argv[1]);
    int cols = atoi(argv[2]);
    auto S = readBin(argv[3], (size_t)rows * cols);

    float *Sd, *Pd;
    size_t bytes = (size_t)rows * cols * sizeof(float);
    CUDA_CHECK(cudaMalloc(&Sd, bytes));
    CUDA_CHECK(cudaMalloc(&Pd, bytes));
    CUDA_CHECK(cudaMemcpy(Sd, S.data(), bytes, cudaMemcpyHostToDevice));

    int threads = 128;
    int blocks = (rows + threads - 1) / threads;
    onlineSoftmaxRows<<<blocks, threads>>>(Sd, Pd, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> P((size_t)rows * cols);
    CUDA_CHECK(cudaMemcpy(P.data(), Pd, bytes, cudaMemcpyDeviceToHost));
    writeBin(argv[4], P);

    cudaFree(Sd); cudaFree(Pd);
    return 0;
}
