#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <vector>
#include "cuda_utils.cuh"

__global__ void matmulQK(const float* Q, const float* K, float* S,
                          int N, int D, float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // query index i
    int col = blockIdx.x * blockDim.x + threadIdx.x; // key index j
    if (row < N && col < N) {
        float acc = 0.0f;
        for (int d = 0; d < D; d++) {
            acc += Q[row * D + d] * K[col * D + d];
        }
        S[row * N + col] = acc * scale;
    }
}

__global__ void softmaxRows(const float* S, float* P, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += expf(S[row * N + j]);
        }
        for (int j = 0; j < N; j++) {
            P[row * N + j] = expf(S[row * N + j]) / sum;
        }
    }
}

__global__ void matmulPV(const float* P, const float* V, float* O,
                          int N, int D) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // output row i
    int col = blockIdx.x * blockDim.x + threadIdx.x; // output col d
    if (row < N && col < D) {
        float acc = 0.0f;
        for (int j = 0; j < N; j++) {
            acc += P[row * N + j] * V[j * D + col];
        }
        O[row * D + col] = acc;
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
    if (argc != 7) {
        fprintf(stderr, "Usage: %s N D q.bin k.bin v.bin o.bin\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    int D = atoi(argv[2]);

    auto Q = readBin(argv[3], (size_t)N * D);
    auto K = readBin(argv[4], (size_t)N * D);
    auto V = readBin(argv[5], (size_t)N * D);

    float *Qd, *Kd, *Vd, *Sd, *Pd, *Od;
    CUDA_CHECK(cudaMalloc(&Qd, (size_t)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Kd, (size_t)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Vd, (size_t)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Sd, (size_t)N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Pd, (size_t)N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Od, (size_t)N * D * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(Qd, Q.data(), (size_t)N * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Kd, K.data(), (size_t)N * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Vd, V.data(), (size_t)N * D * sizeof(float), cudaMemcpyHostToDevice));

    float scale = 1.0f / sqrtf((float)D);

    dim3 block2d(16, 16);
    dim3 gridQK((N + 15) / 16, (N + 15) / 16);
    matmulQK<<<gridQK, block2d>>>(Qd, Kd, Sd, N, D, scale);
    CUDA_CHECK(cudaGetLastError());

    int threads1d = 128;
    int blocks1d = (N + threads1d - 1) / threads1d;
    softmaxRows<<<blocks1d, threads1d>>>(Sd, Pd, N);
    CUDA_CHECK(cudaGetLastError());

    dim3 gridPV((D + 15) / 16, (N + 15) / 16);
    matmulPV<<<gridPV, block2d>>>(Pd, Vd, Od, N, D);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> O((size_t)N * D);
    CUDA_CHECK(cudaMemcpy(O.data(), Od, (size_t)N * D * sizeof(float), cudaMemcpyDeviceToHost));
    writeBin(argv[6], O);

    cudaFree(Qd); cudaFree(Kd); cudaFree(Vd);
    cudaFree(Sd); cudaFree(Pd); cudaFree(Od);
    return 0;
}
