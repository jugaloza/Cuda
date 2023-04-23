
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#define N 50000

__global__ void addWithCUDA(int* d_a, int* d_b, int* d_c)
{
    int threadId = threadIdx.x + (blockDim.x * blockIdx.x);

    while (threadId < N)
    {
        d_c[threadId] = d_a[threadId] + d_b[threadId];
        threadId += blockDim.x * gridDim.x;
    }
}

int main()
{
    
    int* h_a, * h_b, * h_c;

    int* d_a0, *d_a1, *d_b0, *d_b1;
    int* d_c0, * d_c1;

    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "CUDASetDevice failed. " << std::endl;
        return -1;
    }

    cudaStream_t stream0, stream1;

    cudaStatus = cudaStreamCreate(&stream0);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "cudaStreamCreate failed. " << std::endl;
        return -1;
    }
    
    cudaStatus = cudaStreamCreate(&stream1);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "cudaStreamCreate failed. " << std::endl;
        return -1;
    }

    cudaEvent_t e_start, e_stop;

    cudaStatus = cudaEventCreate(&e_start);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "cudaEventCreate failed. " << std::endl;
        return -1;
    }

    cudaStatus = cudaEventCreate(&e_stop);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "cudaEventCreate failed. " << std::endl;
        return -1;
    }

    cudaStatus = cudaEventRecord(e_start, 0);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "CudaEventRecord failed. " << std::endl;
        return -1;
    }

    cudaStatus = cudaHostAlloc((void**)&h_a, 2 * N * sizeof(int), cudaHostAllocDefault);
    
    if (cudaStatus != cudaSuccess)
    {
        std::cout << "cudaHostAlloc failed. " << std::endl;
        return -1;
    }

    cudaStatus = cudaHostAlloc((void**)&h_b, 2 * N * sizeof(int), cudaHostAllocDefault);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "cudaHostAlloc failed. " << std::endl;
        return -1;
    }

    cudaStatus = cudaHostAlloc((void**)&h_c, 2 * N * sizeof(int), cudaHostAllocDefault);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "cudaHostAlloc failed. " << std::endl;
        return -1;
    }

    cudaStatus = cudaMalloc((void**)&d_a0, N * sizeof(int));

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "cudaMalloc failed. " << std::endl;
        return -1;
    }

    cudaStatus = cudaMalloc((void**)&d_a1, N * sizeof(int));

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "CudaMalloc failed. " << std::endl;
        return -1;
    }

    cudaStatus = cudaMalloc((void**)&d_b0, N * sizeof(int));

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "CudaMalloc failed. " << std::endl;
        return -1;
    }

    cudaStatus = cudaMalloc((void**)&d_b1, N * sizeof(int));

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "CudaMalloc failed. " << std::endl;
        return -1;
    }

    cudaStatus = cudaMalloc((void**)&d_c0, N * sizeof(int));

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "CudaMalloc failed. " << std::endl;
        return -1;
    }

    cudaStatus = cudaMalloc((void**)&d_c1, N * sizeof(int));

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "CudaMalloc failed. " << std::endl;
        return -1;
    }

    for (int idx = 0; idx < 2 * N; idx++)
    {
        h_a[idx] = 2 * idx * idx;
        h_b[idx] = idx;
    }

    cudaStatus = cudaMemcpyAsync(d_a0, h_a, N * sizeof(int), cudaMemcpyHostToDevice, stream0);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "CudaMemcpyAsync failed. " << std::endl;
        cudaFree(d_a0);
        cudaFree(d_a1);
        cudaFree(d_b0);
        cudaFree(d_b1);
        cudaFree(d_c0);
        cudaFree(d_c1);
        cudaFreeHost(h_a);
        cudaFreeHost(h_b);
        cudaFreeHost(h_c);
        return -1;
    }
    
    cudaStatus = cudaMemcpyAsync(d_a1, h_a + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "CudaMemcpyAsync failed. " << std::endl;
        cudaFree(d_a0);
        cudaFree(d_a1);
        cudaFree(d_b0);
        cudaFree(d_b1);
        cudaFree(d_c0);
        cudaFree(d_c1);
        cudaFreeHost(h_a);
        cudaFreeHost(h_b);
        cudaFreeHost(h_c);
        return -1;
    }

    cudaStatus = cudaMemcpyAsync(d_b0, h_b, N * sizeof(int), cudaMemcpyHostToDevice, stream0);


    if (cudaStatus != cudaSuccess)
    {
        std::cout << "CudaMemcpyAsync failed. " << std::endl;
        cudaFree(d_a0);
        cudaFree(d_a1);
        cudaFree(d_b0);
        cudaFree(d_b1);
        cudaFree(d_c0);
        cudaFree(d_c1);
        cudaFreeHost(h_a);
        cudaFreeHost(h_b);
        cudaFreeHost(h_c);
        return -1;
    }

    cudaStatus = cudaMemcpyAsync(d_b1, h_b + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "CudaMemcpyAsync failed. " << std::endl;
        cudaFree(d_a0);
        cudaFree(d_a1);
        cudaFree(d_b0);
        cudaFree(d_b1);
        cudaFree(d_c0);
        cudaFree(d_c1);
        cudaFreeHost(h_a);
        cudaFreeHost(h_b);
        cudaFreeHost(h_c);
        return -1;
    }

    addWithCUDA << < 512, 512, 0, stream0 >> > (d_a0, d_b0, d_c0);
    addWithCUDA << < 512, 512, 0, stream1 >> > (d_a1, d_b1, d_c1);

    cudaStatus = cudaMemcpyAsync(h_c, d_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream0);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "CudaMemcpyAsync failed. " << std::endl;
        cudaFree(d_a0);
        cudaFree(d_a1);
        cudaFree(d_b0);
        cudaFree(d_b1);
        cudaFree(d_c0);
        cudaFree(d_c1);
        cudaFreeHost(h_a);
        cudaFreeHost(h_b);
        cudaFreeHost(h_c);
        return -1;
    }

    cudaStatus = cudaMemcpyAsync(h_c + N, d_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream1);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "CudaMemcpyAsync failed. " << std::endl;
        cudaFree(d_a0);
        cudaFree(d_a1);
        cudaFree(d_b0);
        cudaFree(d_b1);
        cudaFree(d_c0);
        cudaFree(d_c1);
        cudaFreeHost(h_a);
        cudaFreeHost(h_b);
        cudaFreeHost(h_c);
        return -1;
    }

    cudaStatus = cudaDeviceSynchronize();

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "CudaMemcpyAsync failed. " << std::endl;
        cudaFree(d_a0);
        cudaFree(d_a1);
        cudaFree(d_b0);
        cudaFree(d_b1);
        cudaFree(d_c0);
        cudaFree(d_c1);
        cudaFreeHost(h_a);
        cudaFreeHost(h_b);
        cudaFreeHost(h_c);
        return -1;
    }

    cudaStatus = cudaStreamSynchronize(stream0);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "CudaMemcpyAsync failed. " << std::endl;
        cudaFree(d_a0);
        cudaFree(d_a1);
        cudaFree(d_b0);
        cudaFree(d_b1);
        cudaFree(d_c0);
        cudaFree(d_c1);
        cudaFreeHost(h_a);
        cudaFreeHost(h_b);
        cudaFreeHost(h_c);
        return -1;
    }

    cudaStatus = cudaStreamSynchronize(stream1);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "CudaMemcpyAsync failed. " << std::endl;
        cudaFree(d_a0);
        cudaFree(d_a1);
        cudaFree(d_b0);
        cudaFree(d_b1);
        cudaFree(d_c0);
        cudaFree(d_c1);
        cudaFreeHost(h_a);
        cudaFreeHost(h_b);
        cudaFreeHost(h_c);
        return -1;
    }

    cudaStatus  = cudaEventRecord(e_stop, 0);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "CudaMemcpyAsync failed. " << std::endl;
        cudaFree(d_a0);
        cudaFree(d_a1);
        cudaFree(d_b0);
        cudaFree(d_b1);
        cudaFree(d_c0);
        cudaFree(d_c1);
        cudaFreeHost(h_a);
        cudaFreeHost(h_b);
        cudaFreeHost(h_c);
        return -1;
    }

    cudaStatus = cudaEventSynchronize(e_stop);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "CudaMemcpyAsync failed. " << std::endl;
        cudaFree(d_a0);
        cudaFree(d_a1);
        cudaFree(d_b0);
        cudaFree(d_b1);
        cudaFree(d_c0);
        cudaFree(d_c1);
        cudaFreeHost(h_a);
        cudaFreeHost(h_b);
        cudaFreeHost(h_c);
        return -1;
    }

    float elapsed_time;

    cudaEventElapsedTime(&elapsed_time, e_start, e_stop);

    std::cout << "Elapsed Time : " << elapsed_time << " ms " << std::endl;

    bool correctSum = 1;

    for (int idx = 0; idx < 2 * N; idx++)
    {
        if (h_a[idx] + h_b[idx] != h_c[idx])
        {
            correctSum = 0;
        }
    }
    
    if (!correctSum)
    {
        std::cout << "GPU sum is not computed correctly.  " << std::endl;
    }
    else
    {
       std::cout << "GPU sum is  computed correctly.  " << std::endl;
    }

    cudaFree(d_a0);
    cudaFree(d_a1);
    cudaFree(d_b0);
    cudaFree(d_a1);
    cudaFree(d_c0);
    cudaFree(d_c1);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    return 0;
}

