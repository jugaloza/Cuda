
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <format>
#include <string>

#define M 8
#define N 8


__global__ void transposeMatrixCuda(int* arr, int* outputArr)
{
    int indexY = blockDim.y * blockIdx.y + threadIdx.y;
    int indexX = blockDim.x * blockDim.x + threadIdx.x;

    int Index = indexY * N + indexX;
    int transposeIndex = indexX * M + indexY;

    outputArr[transposeIndex] = arr[Index];

   
}

int main()
{
    const int totalSize = M * N;
    
    //int arr[M][N] = { 0 };
    //int res[N][M];


    int* arr;
    int* res;

    arr = (int*)malloc(totalSize * sizeof(int));
    

    //filling array
    for (int i = 0; i < totalSize; i++)
    {
        arr[i] = std::rand();
    }

    res = (int*)malloc(totalSize * sizeof(int));

    int* d_inpArray;
    int* d_outputArray;

    cudaError_t errorStatus;

    errorStatus = cudaSetDevice(0);

    if (errorStatus != cudaSuccess)
    {
        std::cout << "Unable to find CUDA capable device " << std::endl;
        return -1;
    }


    errorStatus = cudaMalloc((void**)&d_inpArray, sizeof(int) * totalSize);

    if (errorStatus != cudaSuccess)
    {
        std::cout << " Unable to allocate memory on GPU device for input Array" << std::endl;
        return -1;
    }

    errorStatus = cudaMalloc((void**)&d_outputArray, sizeof(int) * totalSize);

    if (errorStatus != cudaSuccess)
    {
        std::cout << " Unable to allocate memory on GPU device for output Array" << std::endl;
        return -1;
    }

    errorStatus = cudaMemcpy(d_inpArray, arr, sizeof(int) * totalSize, cudaMemcpyHostToDevice);

    if (errorStatus != cudaSuccess)
    {
        std::cout << "Unable to copy content of input array from host to device " << std::endl;
        return -1;
    }
    
    //Transpose matrix
    dim3 numThreads(4, 4);
    dim3 numBlocks(2, 2);

    transposeMatrixCuda << < numBlocks, numThreads >> > (d_inpArray, d_outputArray);

    errorStatus = cudaDeviceSynchronize();

    if (errorStatus != cudaSuccess)
    {
        std::cout << " Error while executing kernel : " << cudaGetErrorString(errorStatus) << std::endl;
        cudaFree(d_inpArray);
        cudaFree(d_outputArray);
        return -1;
    }

    errorStatus = cudaMemcpy(res, d_outputArray, sizeof(int) * totalSize, cudaMemcpyDeviceToHost);

    if (errorStatus != cudaSuccess)
    {
        std::cout << " Unable to copy content of resultant array from device to host " << std::endl;
        cudaFree(d_inpArray);
        cudaFree(d_outputArray);
        return -1;
    }
    
    //printing result

    std::cout << "After transposing matrix " << std::endl;

    for (int idx = 0; idx < totalSize; idx++)
    {
        std::cout << res[idx] << std::endl;
    }

    errorStatus = cudaDeviceReset();

    if (errorStatus != cudaSuccess)
    {
        std::cout << "Failed to reset GPU device  " << std::endl;
        cudaFree(d_inpArray);
        cudaFree(d_outputArray);
        return -1;
    }

    cudaFree(d_inpArray);
    cudaFree(d_outputArray);

    return 0;
}

