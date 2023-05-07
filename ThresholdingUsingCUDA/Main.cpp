#include "opencv2/opencv.hpp"
#include "cuda_runtime.h"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudafilters.hpp"

#define GPU_UNIFIED_MEM

int main()
{
	cv::Mat img = cv::imread("D:\\Opencv_CUDA\\images\\cameraman.jpg", cv::IMREAD_UNCHANGED);

#ifdef GPU_UNIFIED_MEM

	cudaError_t errorStatus;

	void* unique_ptr;
	void* unique_ptr1;
	
	auto frame_size = img.rows * img.cols * 3;
	auto frame_size_gray = img.rows * img.cols;
	
	errorStatus = cudaMallocManaged(&unique_ptr, frame_size);

	if (errorStatus != cudaSuccess)
	{
		std::cout << "Unable to allocate unified memory" << std::endl;
		return -1;
	}

	errorStatus = cudaMallocManaged(&unique_ptr1, frame_size_gray);

	if (errorStatus != cudaSuccess)
	{
		std::cout << "Unable to allocate unified memory" << std::endl;
		return -1;
	}

	cv::Mat h_img(img.rows, img.cols, img.type(), unique_ptr);

	cv::cuda::GpuMat d_img(img.rows, img.cols, img.type(), unique_ptr);

	cv::cuda::GpuMat d_resultImg(img.rows, img.cols, CV_8UC1, unique_ptr1);

	cv::Mat h_resultImg(img.rows, img.cols, CV_8UC1, unique_ptr1);

	//cv::cuda::GpuMat d_img, d_resultImg;
	img.copyTo(h_img);

	auto start = cv::getTickCount();
	//d_img.upload(img);

	cv::cuda::cvtColor(d_img, d_resultImg, cv::COLOR_BGR2GRAY);

	cv::cuda::threshold(d_resultImg, d_resultImg, 127, 255, cv::THRESH_BINARY);

	cv::Ptr<cv::cuda::Filter> filter_3x3;

	cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
	filter_3x3 = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, CV_8UC1, element);
	
	filter_3x3->apply(d_resultImg, d_resultImg);

	auto end = cv::getTickCount();

	auto totalTime = (end - start) / cv::getTickFrequency();

	std::cout << "Time taken on UNIFIED mem : " << totalTime << " ms " << std::endl;
	
	cv::imshow("Resultant Image", h_resultImg);
	cv::waitKey(0);
	cv::destroyAllWindows();


	cudaFree(unique_ptr);
	cudaFree(unique_ptr1);

#else

	cv::cuda::GpuMat d_img, d_resultImg;

	cudaDeviceReset();
	cv::Mat h_resultImg;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
	cv::Ptr<cv::cuda::Filter> filter_3x3;

	auto start = cv::getTickCount();
	d_img.upload(img);
	cv::cuda::cvtColor(d_img, d_img, cv::COLOR_BGR2GRAY);

	cv::cuda::threshold(d_img, d_resultImg, 127, 255, cv::THRESH_BINARY);

	

	filter_3x3 = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, CV_8UC1, element);

	filter_3x3->apply(d_resultImg, d_resultImg);

	d_resultImg.download(h_resultImg);

	auto end = cv::getTickCount();

	auto totalTime = (end - start) / cv::getTickFrequency();

	std::cout << "Time taken on pageable memory : " << totalTime << " ms " << std::endl;
	
#endif
	std::cin.get();
}