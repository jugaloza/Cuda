
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#define NUM_THREADS 10000

#define BLOCK_WIDTH 1000
#define SIZE 10

__global__ void gpu_increment(int* d_inp)
{
    int threadId = threadIdx.x + (blockDim.x * blockIdx.x);

    threadId = threadId % SIZE;

    d_inp[threadId] += 1;
}

__global__ void gpu_increment_atomic(int* d_inp)
{
    int threadId = threadIdx.x + (blockDim.x * blockIdx.x);

    threadId = threadId % SIZE;

    atomicAdd(&d_inp[threadId], 1);

}

int main()
{
    int arr[SIZE] = { 0 };

    int* d_inp;

    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Unable to find CUDA capable device. " << std::endl;
        return -1;
    }

    cudaStatus = cudaMalloc((void**)&d_inp, sizeof(int) * SIZE);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Unable to allocate memory on device. " << std::endl;
        return -1;
    }

    cudaStatus = cudaMemset(d_inp, 0, sizeof(int) * SIZE);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "CudaMemSet failed for input array. " << std::endl;
        cudaFree(d_inp);
        return -1;
    }

    //gpu_increment << < NUM_THREADS / BLOCK_WIDTH, BLOCK_WIDTH >> > (d_inp);
    gpu_increment_atomic << < NUM_THREADS / BLOCK_WIDTH, BLOCK_WIDTH >> > (d_inp);

    cudaStatus = cudaGetLastError();

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Kernel execution failed. " << std::endl;
        cudaFree(d_inp);
        return -1;
    }

    cudaStatus = cudaDeviceSynchronize();

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "CudaDeviceSynchronize failed. " << std::endl;
        cudaFree(d_inp);
        return -1;
    }

    cudaStatus = cudaMemcpy(arr, d_inp, sizeof(int) * SIZE, cudaMemcpyDeviceToHost);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "CudaMemcpy failed. " << std::endl;
        cudaFree(d_inp);
        return -1;
    }

    for (int idx = 0; idx < SIZE; idx++)
    {
        std::cout << arr[idx] << std::endl;
    }

    cudaFree(d_inp);

    
    
    return 0;
}
