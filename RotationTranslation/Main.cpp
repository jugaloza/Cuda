#include "opencv2/opencv.hpp"
#include "cuda_runtime.h"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"

#define UNIFIED_MEM

int main()
{
	cv::Mat img = cv::imread("D:\\Opencv_CUDA\\images\\cameraman.jpg", cv::IMREAD_COLOR);
	
	
#ifndef UNIFIED_MEM
	cv::cuda::GpuMat d_img1,d_result1,d_result2;


	cv::Mat h_result1, h_result2;

	cv::Mat trans_mat = (cv::Mat_<double>(2, 3) << 1, 0, 70, 0, 1, 50);

	cv::Mat rotat_mat = cv::getRotationMatrix2D(cv::Point2f(img.cols / 2, img.rows / 2), 90, 1.0);

	auto start = cv::getTickCount();
	
	d_img1.upload(img);
	
	cv::cuda::warpAffine(d_img1, d_result1, trans_mat, d_img1.size());

	cv::cuda::warpAffine(d_img1, d_result2, rotat_mat, d_img1.size());

	d_result1.download(h_result1);

	d_result2.download(h_result2);

	auto end = cv::getTickCount();

	auto totalTime = (end - start) / cv::getTickFrequency();

	std::cout << "Time taken to perform rotation and translation on GPU : " << totalTime << " ms " << std::endl;
#else

	void* img_ptr;
	void* result1_ptr;
	void* result2_ptr;

	cv::Mat finalResultMat1, finalResultMat2;
	auto frameSize = img.rows * img.cols * 3;

	cudaError_t errorStatus;

	errorStatus = cudaMallocManaged(&img_ptr, frameSize);

	if (errorStatus != cudaSuccess)
	{
		std::cout << cudaGetErrorString(errorStatus) << std::endl;
		cudaFree(img_ptr);
		return -1;
	}

	errorStatus = cudaMallocManaged(&result1_ptr,frameSize);

	if (errorStatus != cudaSuccess)
	{
		std::cout << cudaGetErrorString(errorStatus) << std::endl;
		cudaFree(img_ptr);
		cudaFree(result1_ptr);
		return -1;
	}

	errorStatus = cudaMallocManaged(&result2_ptr, frameSize);

	if (errorStatus != cudaSuccess)
	{
		std::cout << cudaGetErrorString(errorStatus) << std::endl;
		cudaFree(img_ptr);
		cudaFree(result1_ptr);
		cudaFree(result2_ptr);
		return -1;
	}
	
	cv::cuda::GpuMat d_img(img.rows, img.cols, img.type(), img_ptr);

	cv::cuda::GpuMat d_result1(img.rows, img.cols, img.type(), result1_ptr);

	cv::cuda::GpuMat d_result2(img.rows, img.cols, img.type(), result2_ptr);

	cv::Mat h_result1(img.rows, img.cols, img.type(), result1_ptr);
	cv::Mat h_result2(img.rows, img.cols, img.type(), result2_ptr);

	auto start = cv::getTickCount();

	d_img.upload(img);

	cv::Mat rotat_mat = cv::getRotationMatrix2D(cv::Point2f(d_img.cols / 2, d_img.rows / 2), 90, 1.0);
	
	cv::Mat trans_mat = (cv::Mat_<double>(2,3) << 1, 0, 70, 0, 1, 50);

	cv::cuda::warpAffine(d_img, d_result1, rotat_mat, d_img.size());

	cv::cuda::warpAffine(d_img, d_result2, trans_mat, d_img.size());


	auto end = cv::getTickCount();

	auto totalTime = (end - start) / cv::getTickFrequency();

	std::cout << "Time taken to perform rotation and translation on GPU  using UNIFIED Memory : " << totalTime << " ms " << std::endl;


	

	h_result1.copyTo(finalResultMat1);
	h_result2.copyTo(finalResultMat2);
	

	cudaFree(img_ptr);
	cudaFree(result1_ptr);
	cudaFree(result2_ptr);

	

#endif
	
	cv::imshow("Image2", finalResultMat1);
	cv::waitKey(0);
	cv::destroyAllWindows();
	std::cin.get();

}