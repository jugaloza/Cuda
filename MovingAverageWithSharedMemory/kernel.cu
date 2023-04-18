
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <iostream>


__global__ void movingAverageCUDA(int* d_inp, float* d_out, int N)
{
    int threadId = threadIdx.x;

    __shared__ int sh_array[10];

    sh_array[threadId] = d_inp[threadId];

    __syncthreads();

    int sum = 0;
    float average = 0.0f;

    for (int idx = 0; idx < threadId; idx++)
    {
        sum += sh_array[idx];
    }

    average = sum / (threadId + 1.0f);

    d_out[threadId] = average;

    sh_array[threadId] = average;

}


int main()
{
    
    const int N = 10;
    
    int arr[N] = { 0 };
    float out_arr[N] = { 0.0f };

    int* d_inp;
    float* d_out;

    //fill array
    for (int idx = 0; idx < N; idx++)
    {
        arr[idx] = idx;
    }

    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Unable to find CUDA capable device. Please check if this system has CUDA capable device" << std::endl;
        return -1;
    }

    cudaStatus = cudaDeviceReset();

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "CUDA Device Reset failed. " << std::endl;
        return -1;
    }

    cudaStatus = cudaMalloc((void**)&d_inp, sizeof(int) * N);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "CUDA Malloc for input array on device failed. " << std::endl;
        return -1;
    }

    cudaStatus = cudaMalloc((void**)&d_out, sizeof(float) * N);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Unable to allocate space on device for output array. " << std::endl;
        cudaFree(d_inp);
        return -1;
    }

    cudaStatus = cudaMemcpy(d_inp, arr, sizeof(int) * N, cudaMemcpyHostToDevice);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Unable to copy content of input array from host to device. " << std::endl;
        cudaFree(d_inp);
        cudaFree(d_out);
        return -1;
    }

    movingAverageCUDA << <1, 10 >> > (d_inp, d_out, N);

    cudaStatus = cudaGetLastError();

    if (cudaStatus != cudaSuccess)
    {
        std::cout << " Error while executing kernel code on GPU. " << std::endl;
        cudaFree(d_inp);
        cudaFree(d_out);
        return -1;
    }

    cudaStatus = cudaDeviceSynchronize();

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Unable to sync cuda device. " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_inp);
        cudaFree(d_out);
        return -1;
    }

    cudaStatus = cudaMemcpy(out_arr, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "unable to copy output from device to host. " << std::endl;
        cudaFree(d_inp);
        cudaFree(d_out);
        return -1;
    }

    //printing output array

    for (int idx = 0; idx < N; idx++)
    {
        std::cout << out_arr[idx] << std::endl;
    }

    cudaFree(d_inp);
    cudaFree(d_out);

    cudaStatus = cudaDeviceReset();

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Unable to reset cuda Device. " << cudaGetErrorString(cudaStatus) << std::endl;
        return -1;
    }
    return 0;
}

