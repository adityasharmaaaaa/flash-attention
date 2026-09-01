#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <vector>
#include <limits>
#include "cuda_utils.cuh"

// unstable: no max-subtraction (this is Stage 1's softmaxRows, renamed for clarity)
__global__ void unstableSoftmaxRows(const float* S, float* P, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) sum += expf(S[i * cols + j]);
        for (int j = 0; j < cols; j++) P[i * cols + j] = expf(S[i * cols + j]) / sum;
    }
}

// stable: max-subtraction
__global__ void stableSoftmaxRows(const float* S, float* P, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) {
        float m = -INFINITY;
        for (int j = 0; j < cols; j++) m = fmaxf(m, S[i * cols + j]);

        float l = 0.0f;
        for (int j = 0; j < cols; j++) l += expf(S[i * cols + j] - m);

        for (int j = 0; j < cols; j++) P[i * cols + j] = expf(S[i * cols + j] - m) / l;
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
        fprintf(stderr, "Usage: %s rows cols s.bin out_prefix\n", argv[0]);
        return 1;
    }
    int rows = atoi(argv[1]);
    int cols = atoi(argv[2]);
    auto S = readBin(argv[3], (size_t)rows * cols);

    float *Sd, *Pu, *Ps;
    size_t bytes = (size_t)rows * cols * sizeof(float);
    CUDA_CHECK(cudaMalloc(&Sd, bytes));
    CUDA_CHECK(cudaMalloc(&Pu, bytes));
    CUDA_CHECK(cudaMalloc(&Ps, bytes));
    CUDA_CHECK(cudaMemcpy(Sd, S.data(), bytes, cudaMemcpyHostToDevice));

    int threads = 128;
    int blocks = (rows + threads - 1) / threads;
    unstableSoftmaxRows<<<blocks, threads>>>(Sd, Pu, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    stableSoftmaxRows<<<blocks, threads>>>(Sd, Ps, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> outU((size_t)rows * cols), outS((size_t)rows * cols);
    CUDA_CHECK(cudaMemcpy(outU.data(), Pu, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(outS.data(), Ps, bytes, cudaMemcpyDeviceToHost));

    std::string prefix = argv[4];
    writeBin((prefix + "_unstable.bin").c_str(), outU);
    writeBin((prefix + "_stable.bin").c_str(), outS);

    cudaFree(Sd); cudaFree(Pu); cudaFree(Ps);
    return 0;
}
