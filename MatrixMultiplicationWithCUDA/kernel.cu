
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#define BLOCK_WIDTH 2


__global__ void gpuMatrixMultiplication(int* d_a, int* d_b, int* d_c, int size)
{
    int row = threadIdx.y + (blockDim.y + blockIdx.y);
    int col = threadIdx.x + (blockDim.x + blockIdx.x);

    for (int k = 0; k < size; k++)
    {
        d_c[row * size + col] += d_a[row * size + k] * d_b[k * size + col];
    }
}

int main()
{
    int size = 4;

    int h_a[4][4] = { 0 };
    int h_b[4][4] = { 0 };
    int h_c[4][4] = { 0 };
    

    int* d_a, * d_b, * d_c;

    //int* h_c;

    //h_c = (int*)malloc(size * size * sizeof(int));

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            h_a[i][j] = i;
            h_b[i][j] = j;
        }
    }

    cudaError_t cudaStatus;

    cudaEvent_t e_start,e_stop;

    cudaEventCreate(&e_start);
    cudaEventCreate(&e_stop);
    cudaEventRecord(e_start, 0);

    cudaStatus = cudaSetDevice(0);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Unable to find CUDA capable device. " << std::endl;
        return -1;
    }

    cudaStatus = cudaMalloc((void**)&d_a, size * size * sizeof(int));

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Unable to allocate memory on device. " << std::endl;
        return -1;
    }

    cudaStatus = cudaMalloc((void**)&d_b, size * size * sizeof(int));

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Unable to allocate memory on device. " << std::endl;
        cudaFree(d_a);
        return -1;
    }

    cudaStatus = cudaMalloc((void**)&d_c, size * size * sizeof(int));

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Unable to allocate memory on device. " << std::endl;
        cudaFree(d_b);
        cudaFree(d_a);
        return -1;
    }

    cudaStatus = cudaMemcpy(d_a, h_a, size * size * sizeof(int), cudaMemcpyHostToDevice);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "CudaMemcpy failed. " << std::endl;
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return -1;
    }

    cudaStatus = cudaMemcpy(d_b, h_b, size * size * sizeof(int), cudaMemcpyHostToDevice);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "CudaMemcpy failed. " << std::endl;
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return -1;
    }

    dim3 numBlocks(size / BLOCK_WIDTH, size / BLOCK_WIDTH);
    dim3 numThreads(BLOCK_WIDTH, BLOCK_WIDTH);
    gpuMatrixMultiplication << < numBlocks, numThreads >> > (d_a, d_b, d_c,size);

    cudaStatus = cudaGetLastError();

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Kernel execution failed. " << std::endl;
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return -1;
    }

    cudaStatus = cudaDeviceSynchronize();

    if (cudaStatus != cudaSuccess)
    {
        std::cout << " DEvice sync failed. " << std::endl;
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return -1;
    }

    cudaStatus = cudaMemcpy(h_c, d_c, size * size * sizeof(int), cudaMemcpyDeviceToHost);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "CudaMemcpy failed. " << std::endl;
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return -1;
    }

    cudaEventRecord(e_stop, 0);
    cudaEventSynchronize(e_stop);
    
    float elapsed_time;

    cudaEventElapsedTime(&elapsed_time, e_start, e_stop);

    std::cout << "Elapsed Time : " << elapsed_time <<  " ms " << std::endl;


    std::cout << "Output of mult" << std::endl;

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            std::cout << h_c[i][j] << std::endl;
        }
    }
    std::cin.get();
    return 0;
}
