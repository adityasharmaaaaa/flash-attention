#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <vector>
#include "cuda_utils.cuh"

// ---- forward kernels (duplicated from Stage 1 -- see note in docs) ----

__global__ void matmulQK(const float* Q, const float* K, float* S,
                          int N, int D, float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float acc = 0.0f;
        for (int d = 0; d < D; d++) acc += Q[row * D + d] * K[col * D + d];
        S[row * N + col] = acc * scale;
    }
}

__global__ void softmaxRows(const float* S, float* P, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) sum += expf(S[row * N + j]);
        for (int j = 0; j < N; j++) P[row * N + j] = expf(S[row * N + j]) / sum;
    }
}

// ---- backward kernels ----

// dP[i][j] = sum_d dO[i][d] * V[j][d]      (dP = dO @ V^T)
__global__ void compute_dP(const float* dO, const float* V, float* dP,
                            int N, int D) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N) {
        float acc = 0.0f;
        for (int d = 0; d < D; d++) acc += dO[i * D + d] * V[j * D + d];
        dP[i * N + j] = acc;
    }
}

// rowsum[i] = sum_j P[i][j] * dP[i][j]
__global__ void rowsum_kernel(const float* P, const float* dP, float* rowsum, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float acc = 0.0f;
        for (int j = 0; j < N; j++) acc += P[i * N + j] * dP[i * N + j];
        rowsum[i] = acc;
    }
}

// dS[i][j] = P[i][j] * (dP[i][j] - rowsum[i])
__global__ void compute_dS(const float* P, const float* dP, const float* rowsum,
                            float* dS, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N) {
        dS[i * N + j] = P[i * N + j] * (dP[i * N + j] - rowsum[i]);
    }
}

// dQ[i][d] = scale * sum_j dS[i][j] * K[j][d]
__global__ void compute_dQ(const float* dS, const float* K, float* dQ,
                            int N, int D, float scale) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && d < D) {
        float acc = 0.0f;
        for (int j = 0; j < N; j++) acc += dS[i * N + j] * K[j * D + d];
        dQ[i * D + d] = acc * scale;
    }
}

// dK[j][d] = scale * sum_i dS[i][j] * Q[i][d]   -- NOTE: transposed access on dS
__global__ void compute_dK(const float* dS, const float* Q, float* dK,
                            int N, int D, float scale) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N && d < D) {
        float acc = 0.0f;
        for (int i = 0; i < N; i++) acc += dS[i * N + j] * Q[i * D + d];
        dK[j * D + d] = acc * scale;
    }
}

// dV[j][d] = sum_i P[i][j] * dO[i][d]           -- NOTE: transposed access on P
__global__ void compute_dV(const float* P, const float* dO, float* dV,
                            int N, int D) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N && d < D) {
        float acc = 0.0f;
        for (int i = 0; i < N; i++) acc += P[i * N + j] * dO[i * D + d];
        dV[j * D + d] = acc;
    }
}

// ---- host I/O helpers ----

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
    if (argc != 10) {
        fprintf(stderr, "Usage: %s N D q.bin k.bin v.bin dO.bin dq.bin dk.bin dv.bin\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    int D = atoi(argv[2]);

    auto Q  = readBin(argv[3], (size_t)N * D);
    auto K  = readBin(argv[4], (size_t)N * D);
    auto V  = readBin(argv[5], (size_t)N * D);
    auto dO = readBin(argv[6], (size_t)N * D);

    float *Qd, *Kd, *Vd, *dOd, *Sd, *Pd, *dPd, *rowsumd, *dSd, *dQd, *dKd, *dVd;
    size_t ND = (size_t)N * D, NN = (size_t)N * N;

    CUDA_CHECK(cudaMalloc(&Qd, ND * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Kd, ND * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Vd, ND * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOd, ND * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Sd, NN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Pd, NN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dPd, NN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&rowsumd, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dSd, NN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dQd, ND * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dKd, ND * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dVd, ND * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(Qd, Q.data(), ND * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Kd, K.data(), ND * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Vd, V.data(), ND * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dOd, dO.data(), ND * sizeof(float), cudaMemcpyHostToDevice));

    float scale = 1.0f / sqrtf((float)D);
    dim3 block2d(16, 16);
    dim3 gridNN((N + 15) / 16, (N + 15) / 16);
    dim3 gridND((D + 15) / 16, (N + 15) / 16);
    int threads1d = 128;
    int blocks1d = (N + threads1d - 1) / threads1d;

    // forward (recompute P)
    matmulQK<<<gridNN, block2d>>>(Qd, Kd, Sd, N, D, scale);
    CUDA_CHECK(cudaGetLastError());
    softmaxRows<<<blocks1d, threads1d>>>(Sd, Pd, N);
    CUDA_CHECK(cudaGetLastError());

    // backward
    compute_dP<<<gridNN, block2d>>>(dOd, Vd, dPd, N, D);
    CUDA_CHECK(cudaGetLastError());
    rowsum_kernel<<<blocks1d, threads1d>>>(Pd, dPd, rowsumd, N);
    CUDA_CHECK(cudaGetLastError());
    compute_dS<<<gridNN, block2d>>>(Pd, dPd, rowsumd, dSd, N);
    CUDA_CHECK(cudaGetLastError());
    compute_dQ<<<gridND, block2d>>>(dSd, Kd, dQd, N, D, scale);
    CUDA_CHECK(cudaGetLastError());
    compute_dK<<<gridND, block2d>>>(dSd, Qd, dKd, N, D, scale);
    CUDA_CHECK(cudaGetLastError());
    compute_dV<<<gridND, block2d>>>(Pd, dOd, dVd, N, D);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> dQ(ND), dK(ND), dV(ND);
    CUDA_CHECK(cudaMemcpy(dQ.data(), dQd, ND * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dK.data(), dKd, ND * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dV.data(), dVd, ND * sizeof(float), cudaMemcpyDeviceToHost));

    writeBin(argv[7], dQ);
    writeBin(argv[8], dK);
    writeBin(argv[9], dV);

    cudaFree(Qd); cudaFree(Kd); cudaFree(Vd); cudaFree(dOd);
    cudaFree(Sd); cudaFree(Pd); cudaFree(dPd); cudaFree(rowsumd);
    cudaFree(dSd); cudaFree(dQd); cudaFree(dKd); cudaFree(dVd);
    return 0;
}
