
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

__global__ void cubeKernelCuda(int* d_inp, int* d_out)
{
    unsigned int threadId = threadIdx.x +(blockDim.x*blockIdx.x);

    int val = d_inp[threadId];

    d_out[threadId] = val * val * val;

}


int main()
{
    const int N = 50;

    int h_inp[N] = { 0 };
    int h_out[N] = { 0 };

    //filling array
    for (int idx = 0; idx < N; idx++)
    {
        h_inp[idx] = idx;
    }

    int* d_inp;
    int* d_out;

    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << " Unable to find CUDA capable device. " << cudaGetErrorString(cudaStatus) << std::endl;
        return -1;
    }

    cudaStatus = cudaMalloc((void**)&d_inp, sizeof(int) * N);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "CudaMalloc Failed for input array. " << cudaGetErrorString(cudaStatus) << std::endl;
        return -1;
    }

    cudaStatus = cudaMalloc((void**)&d_out, sizeof(int) * N);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "CudaMalloc Failed for output array. " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_inp);
        return -1;
    }

    cudaStatus = cudaMemcpy(d_inp, h_inp, sizeof(int) * N, cudaMemcpyHostToDevice);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "CudaMemCpy Failed for input array from host to device. " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_inp);
        cudaFree(d_out);
        return -1;
    }

    cubeKernelCuda << <2, 25 >> > (d_inp, d_out);

    cudaStatus = cudaGetLastError();

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Unable to complete kernel call on device. " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_inp);
        cudaFree(d_out);
        return -1;
    }

    cudaStatus = cudaDeviceSynchronize();

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "CudaDeviceSynchronize failed : " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_inp);
        cudaFree(d_out);
        return -1;
    }

    cudaStatus = cudaMemcpy(h_out, d_out, sizeof(int) * N, cudaMemcpyDeviceToHost);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "CudaMemcpy failed for output array. ErrorString : " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_inp);
        cudaFree(d_out);
        return -1;

    }

    //printing output array

    for (int idx = 0; idx < N; idx++)
    {
        std::cout << h_out[idx] << std::endl;
    }
    cudaFree(d_inp);
    cudaFree(d_out);

    cudaStatus = cudaDeviceReset();

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "cudaDeviceReset failed. Error : " << cudaGetErrorString(cudaStatus) << std::endl;
        return -1;
    }

    return 0;
}

