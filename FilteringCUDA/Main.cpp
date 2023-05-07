#include "opencv2/opencv.hpp"
#include "opencv2/cudafilters.hpp"
#include "cuda_runtime.h"
#include "opencv2/cudaimgproc.hpp"

#define CUDA_GPU

int main()
{
	cv::Mat img = cv::imread("D:\\Opencv_CUDA\\images\\cameraman.jpg", cv::IMREAD_UNCHANGED);

	
	cv::Mat h_resultImg,h_gaussianImg;

#ifdef CUDA_GPU
	cudaError_t errorStatus;

	errorStatus = cudaDeviceReset();

	if (errorStatus != cudaSuccess)
	{
		std::cout << "Unable to reset cuda device" << std::endl;
		return -1;
	}
	cv::cuda::GpuMat d_img, d_finalImg,d_gaussianBlurImg,d_resultImg;
	auto start = cv::getTickCount();

	d_img.upload(img);

	
	cv::cuda::cvtColor(d_img, d_img, cv::COLOR_BGR2GRAY);

	//cv::Ptr<cv::cuda::Filter> filter_3x3;
	cv::Ptr<cv::cuda::Filter> gaussFilter_3x3;
	cv::Ptr<cv::cuda::Filter> laplacianFilter;

	//filter_3x3 = cv::cuda::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(3, 3));
	gaussFilter_3x3 = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(3, 3), 1);
	laplacianFilter = cv::cuda::createLaplacianFilter(CV_8UC1, CV_8UC1, 1);


	//filter_3x3->apply(d_img, d_finalImg);
	gaussFilter_3x3->apply(d_img, d_gaussianBlurImg);
	laplacianFilter->apply(d_gaussianBlurImg, d_resultImg);

	//d_finalImg.download(h_resultImg);

	d_gaussianBlurImg.download(h_gaussianImg);

	d_resultImg.download(h_resultImg);

	auto end = cv::getTickCount();

	auto total_time = (end - start) / cv::getTickFrequency();

	std::cout << "Time taken to apply blurring using GPU " << total_time << " ms " << std::endl;

#else
	cv::Mat gaussBlurImg;
	auto start = cv::getTickCount();


	cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

	cv::boxFilter(img, h_resultImg, -1, cv::Size(3, 3));

	
	cv::GaussianBlur(img, gaussBlurImg, cv::Size(3, 3), 1);

	cv::Laplacian(gaussBlurImg, h_resultImg, -1);

	auto end = cv::getTickCount();

	auto totalTime = (end - start) / cv::getTickFrequency();

	std::cout << "Total time taken by blurring image on CPU is " << totalTime << " ms " << std::endl;

#endif

	cv::imshow("Blurred image", h_resultImg);
	cv::waitKey(0);
	cv::destroyAllWindows();
	std::cin.get();

	
}