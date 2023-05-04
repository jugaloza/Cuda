#include "opencv2/opencv.hpp"
#include "opencv2/cudacodec.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"
#include "cuda_runtime.h"


#define UNIFIED_MEM

int main()
{
	cv::Mat img = cv::imread("D:\\Opencv_CUDA\\images\\cameraman.jpg", cv::IMREAD_UNCHANGED);
	cv::Mat img1 = cv::imread("D:\\Opencv_CUDA\\images\\circles.jpg", cv::IMREAD_UNCHANGED);
	cv::Mat result1;
	
	
#ifdef UNIFIED_MEM
	cv::Mat h_resizedImg(cv::Size(512, 512), img.type());
	cv::Mat h_resizedImg1(cv::Size(512, 512), img1.type());
	cv::Mat h_result1(cv::Size(512,512),img1.type());

	

	void* img_ptr;
	void* img1_ptr;
	void* resizedImg_ptr;
	void* resizedImg1_ptr;
	void* result_ptr;

	cudaError_t errorStatus;

	auto frameSize = img.rows * img.cols * 3;
	auto resizeFrameSize = 512 * 512 * 3;

	errorStatus = cudaMallocManaged(&img_ptr, frameSize);

	if (errorStatus != cudaSuccess)
	{
		std::cout << "abc" << cudaGetErrorString(errorStatus) << std::endl;
		std::cin.get();
		return -1;
	}

	errorStatus = cudaMallocManaged(&img1_ptr, frameSize);

	if (errorStatus != cudaSuccess)
	{
		std::cout << "abc" << cudaGetErrorString(errorStatus) << std::endl;
		std::cin.get();
		return -1;
	}

	errorStatus = cudaMallocManaged(&resizedImg_ptr, resizeFrameSize);

	if (errorStatus != cudaSuccess)
	{
		std::cout << "abc" << cudaGetErrorString(errorStatus) << std::endl;
		std::cin.get();
		return -1;
	}

	errorStatus = cudaMallocManaged(&resizedImg1_ptr, resizeFrameSize);
	if (errorStatus != cudaSuccess)
	{
		std::cout << "abc" << cudaGetErrorString(errorStatus) << std::endl;
		std::cin.get();
		return -1;
	}


	errorStatus = cudaMallocManaged(&result_ptr, resizeFrameSize);

	if (errorStatus != cudaSuccess)
	{
		std::cout << "abc" << cudaGetErrorString(errorStatus) << std::endl;
		std::cin.get();
		return -1;
	}

	cv::cuda::GpuMat d_img(img.rows, img.cols, img.type(), img_ptr);
	cv::cuda::GpuMat d_img1(img1.rows, img1.cols, img1.type(), img1_ptr);

	
	cv::cuda::GpuMat d_resizedImg(512, 512, img.type(), resizedImg_ptr);
	cv::cuda::GpuMat d_resizedImg1(512, 512, img1.type(), resizedImg1_ptr);

	cv::Mat resultMat(512, 512, img.type(), result_ptr);
	cv::cuda::GpuMat d_resultMat(512, 512, img.type(), result_ptr);
	
	auto start = cv::getTickCount();

	d_img.upload(img);
	d_img1.upload(img1);

	cv::cuda::resize(d_img, d_resizedImg, cv::Size(512, 512));
	cv::cuda::resize(d_img1, d_resizedImg1, cv::Size(512, 512));

	

	

	cv::cuda::add(d_resizedImg, d_resizedImg1, d_resultMat);
	
	auto end = cv::getTickCount();

	auto result = (end - start) / cv::getTickFrequency();

	

	


	std::cout << "Time taken on GPU to execute using Unified Memory: " << result << " ms " << std::endl;

	cudaFree(img_ptr);
	cudaFree(img1_ptr);
	cudaFree(resizedImg_ptr);
	cudaFree(resizedImg1_ptr);
	cudaFree(result_ptr);

#else
	cv::cuda::GpuMat d_img, d_img1, d_result;

	cv::Mat resultMat;

	auto start = cv::getTickCount();
	d_img.upload(img);
	d_img1.upload(img1);

	cv::cuda::resize(d_img, d_img, cv::Size(512, 512));
	cv::cuda::resize(d_img1, d_img1, cv::Size(512, 512));

	cv::cuda::add(d_img, d_img1, d_result);

	d_result.download(resultMat);

	auto end = cv::getTickCount();

	auto total = (end - start) / cv::getTickFrequency();

	std::cout << "Time taken on GPU to execute using pageable memory : " << total << " ms " << std::endl;
#endif

	std::cin.get();

	return 0;
}