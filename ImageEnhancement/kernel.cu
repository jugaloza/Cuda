
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <iostream>

#define IMAGE_WIDTH 16
#define IMAGE_HEIGHT 16
#define MAX_PIXEL_INTENSITY 255
#define MIN_PIXEL_INTENSITY 0


__global__ void enhanceImageWithCuda(int* img, int* res_img,int scale, int val)
{
    unsigned int row_idx = (blockDim.y * blockIdx.y) + threadIdx.y;
    unsigned int col_idx = (blockDim.x * blockIdx.x) + threadIdx.x;

    res_img[col_idx * IMAGE_WIDTH + row_idx] = scale * img[col_idx * IMAGE_WIDTH + row_idx] + val;

}

int main()
{

    int img[IMAGE_HEIGHT][IMAGE_WIDTH] = { 0 };
    int resImg[IMAGE_HEIGHT][IMAGE_WIDTH];
    //fill array
    for (int h_idx = 0; h_idx < IMAGE_HEIGHT; h_idx++)
    {
        for (int w_idx = 0; w_idx < IMAGE_WIDTH; w_idx++)
        {
            img[h_idx][w_idx] = std::rand() % MAX_PIXEL_INTENSITY;
        }
    }

    int* d_resImg, * d_img;

    cudaError_t errorStatus;

    errorStatus = cudaSetDevice(0);

    if (errorStatus != cudaSuccess)
    {
        std::cout << "Unable to find CUDA Capable device. please check if cuda capable device is available" << std::endl;
        return -1;
    }

    errorStatus = cudaDeviceReset();

    if (errorStatus != cudaSuccess)
    {
        std::cout << "Cuda Error : " << cudaGetErrorString(errorStatus) << std::endl;
        return -1;
    }

    errorStatus = cudaMalloc((void**)&d_img, sizeof(int) * (IMAGE_HEIGHT * IMAGE_WIDTH));

    if (errorStatus != cudaSuccess)
    {
        std::cout << "Unable to allocate memory on device " << std::endl;
        return -1;
    }

    errorStatus = cudaMalloc((void**)&d_resImg, sizeof(int) * (IMAGE_HEIGHT * IMAGE_WIDTH));

    if (errorStatus != cudaSuccess)
    {
        std::cout << "Unable to allocate memory on device " << std::endl;
        return -1;
    }


    errorStatus = cudaMemcpy(d_img, img, sizeof(int) * (IMAGE_WIDTH * IMAGE_HEIGHT), cudaMemcpyHostToDevice);

    if (errorStatus != cudaSuccess)
    {
        std::cout << "unable to copy content of input image from host to device " << std::endl;
        cudaFree(d_img);
        cudaFree(d_resImg);
        return -1;
    }

    dim3 numThreads(8, 8);
    dim3 numBlocks(IMAGE_WIDTH / numThreads.x, IMAGE_HEIGHT / numThreads.y);

    enhanceImageWithCuda << <numBlocks, numThreads >> > (d_img, d_resImg, 1, 2);

    errorStatus = cudaDeviceSynchronize();

    if (errorStatus != cudaSuccess)
    {
        std::cout << "Error  : " << cudaGetErrorString(errorStatus) << std::endl;
        cudaFree(d_img);
        cudaFree(d_resImg);
        return -1;
    }


    errorStatus = cudaMemcpy(resImg, d_resImg, sizeof(int) * (IMAGE_WIDTH * IMAGE_HEIGHT), cudaMemcpyDeviceToHost);

    if (errorStatus != cudaSuccess)
    {
        std::cout << "Error unable to copy content of device to host " << std::endl;;
        cudaFree(d_img);
        cudaFree(d_resImg);
        return -1;
    }

    cudaFree(d_img);
    cudaFree(d_resImg);

    for (int h_idx = 0; h_idx < IMAGE_HEIGHT; h_idx++)
    {
        for (int w_idx = 0; w_idx < IMAGE_WIDTH; w_idx++)
        {
            std::cout << resImg[h_idx][w_idx] << std::endl;

        }
    }
    
    std::cin.get();

    return 0;
}
