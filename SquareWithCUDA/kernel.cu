
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

__global__ void squareKernel(int* d_inp, int* d_out)
{
    unsigned int threadId = threadIdx.x;

    auto temp = d_inp[threadId];
    d_out[threadId] = temp * temp;

}
int main()
{
    const int N = 5;

    int h_inp[N] = { 0 };
    int h_out[N] = { 0 };

    for (int idx = 0; idx < N; idx++)
    {
        h_inp[idx] = idx + 1;
    }

    int* d_inp;
    int* d_out;

    cudaError_t cudaStatus;


    cudaStatus = cudaSetDevice(0);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Unable to find CUDA capable device " << std::endl;
        return -1;
    }

    cudaStatus = cudaMalloc((void**)&d_inp, sizeof(int) * N);
    
    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Error while allocating memory on device" << cudaGetErrorString(cudaStatus) << std::endl;
        return -1;
    }

    cudaStatus = cudaMalloc((void**)&d_out, sizeof(int) * N);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Error while allocating memory on device for output array : " << cudaGetErrorString(cudaStatus) << std::endl;
        return -1;
    }

    cudaStatus = cudaMemcpy(d_inp, h_inp, sizeof(int) * N, cudaMemcpyHostToDevice);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Unable to copy content of array from host to device for input array  : " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_inp);
        cudaFree(d_out);
        return -1;
    }

    squareKernel <<<1, 5>>> (d_inp, d_out);

    cudaStatus = cudaGetLastError();

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Error thrown by last thread : " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_inp);
        cudaFree(d_out);
        return -1;
    }

    cudaStatus = cudaDeviceSynchronize();

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Unable to sync device and some task has failed while execution : " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_inp);
        cudaFree(d_out);
        return -1;
    }

    cudaStatus = cudaMemcpy(h_out, d_out, sizeof(int) * N, cudaMemcpyDeviceToHost);

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Unable to copy content of output array from device to host. " << std::endl;
        cudaFree(d_inp);
        cudaFree(d_out);
        return -1;
    }

    //printing squares of vector
    for (int idx = 0; idx < N; idx++)
    {
        std::cout << h_out[idx] << std::endl;
        
    }

    cudaFree(d_inp);
    cudaFree(d_out);

    cudaStatus = cudaDeviceReset();

    if (cudaStatus != cudaSuccess)
    {
        std::cout << "Unable to reset states of device " << std::endl;
        return -1;
    }

    std::cin.get();
    return 0;
}
