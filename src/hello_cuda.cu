#include <cstdio>
#include <cmath>
#include "cuda_utils.cuh"

__global__ void vecAddKernel(const float* A, const float* B, float* C, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<N){
        C[idx] = A[idx] + B[idx];
    }
}

int main(){
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    printf("Device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d\n", prop.multiProcessorCount);

    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);
    
    float* A = (float*)malloc(bytes);
    float* B = (float*)malloc(bytes);
    float *C_gpu = (float*)malloc(bytes);
    float *C_cpu = (float*)malloc(bytes);

    for (int i = 0; i < N; i++) {
        A[i] = static_cast<float>(i % 100) * 0.5f;
        B[i] = static_cast<float>(i % 37) * 0.25f;
    }

    float *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc(&A_d,bytes));
    CUDA_CHECK(cudaMalloc(&B_d,bytes));
    CUDA_CHECK(cudaMalloc(&C_d, bytes));

    CUDA_CHECK(cudaMemcpy(A_d,A,bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d,B,bytes,cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N+threads-1)/threads;

    vecAddKernel<<<blocks,threads>>>(A_d,B_d,C_d,N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C_gpu,C_d,bytes,cudaMemcpyDeviceToHost));

    for(int i=0; i<N; i++) C_cpu[i] = A[i] + B[i];

    float maxDiff = 0.0f;
    for(int i=0; i<N; i++){
        maxDiff = fmaxf(maxDiff, fabsf(C_gpu[i]-C_cpu[i]));
    }

    printf("Max difference (CPU vs GPU): %e\n", maxDiff);
    printf(maxDiff < 1e-5f ? "PASS\n" : "FAIL\n");

    free(A); free(B); free(C_gpu); free(C_cpu);
    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
    return 0;
}