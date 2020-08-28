#include <CL/cl.h>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <opencv4/opencv2/opencv.hpp>
//#include <opencv4/opencv2/core/cvstd.hpp>
//#include <opencv4/opencv2/imgproc/imgproc.hpp>
//#include <opencv4/opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
#pragma comment (lib,"OpenCL.lib")

//把文本文件读入一个 string 中
int convertToString(const char *filename, std::string& s)
{
	size_t size;
	char* str;
	std::fstream f(filename, (std::fstream::in | std::fstream::binary));
	if (f.is_open())
	{
		size_t fileSize;
		f.seekg(0, std::fstream::end);
		size = fileSize = (size_t)f.tellg();
		f.seekg(0, std::fstream::beg);
		str = new char[size + 1];
		if (!str)
		{
			f.close();
			return NULL;
		}
		f.read(str, fileSize);
		f.close();
		str[size] = '\0';
		s = str;
		delete[] str;
		return 0;
	}
	printf("Error: Failed to open file %s\n", filename);
	return 1;
}

int main()
{
	// 卷积核
	const int filterWidth = 5;
	const int filterSize = filterWidth*filterWidth;
	const int halfFilterWidth = filterWidth/2;
	float filter[filterSize] =
		/*
		   {// 恒等映射
		   0, 0, 0, 0, 0,
		   0, 0, 0, 0, 0,
		   0, 0, 1, 0, 0,
		   0, 0, 0, 0, 0,
		   0, 0, 0, 0, 0        
		   };    
		 */
	{// 边缘检测
		-3, 0,-1, 0, 2,
		0,-1, 0, 2, 0,
		-1, 0, 4, 0,-1,
		0, 2, 0,-1, 0,
		2, 0,-1, 0,-3,
	};

	// 图片相关        
	Mat image = imread("./bb.jpg");// 读取图片，OpenCV 自动识别文件类型，返回一个 Mat 类
	Mat channel[3];// 分别存放图像的三个通道
	split(image, channel); // 将原图像拆分为三个通道，分别为蓝色、绿色、红色
	size_t imageHeight = image.rows, imageWidth = image.cols;// 获取图像的行数和列数
	cout<<"行："<<imageHeight<<"，列："<<imageWidth<<endl;
	float *imageData = (float*)malloc(sizeof(float)*imageHeight*imageWidth);//分配内存

	cl_int status;
	cl_uint numPlatforms=0;
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	cl_platform_id* platforms = new cl_platform_id[numPlatforms];
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);

	cl_uint numDevices=0;
	cl_device_id* devices=NULL;
	status=clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
	devices=(cl_device_id*) malloc(numDevices*sizeof(cl_device_id));
	status=clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);

	cl_context context=NULL;
	context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);

	cl_command_queue commandQueue;
	commandQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &status);
	if (commandQueue == NULL) perror("Failed to create commandQueue for device 0.");

	// 设置 image 数据描述符,https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/cl_image_desc.html
	//typedef struct _cl_image_desc {
	//	cl_mem_object_type image_type;
	//	size_t image_width;
	//	size_t image_height;
	//	size_t image_depth;
	//	size_t image_array_size;
	//	size_t image_row_pitch;
	//	size_t image_slice_pitch;
	//	cl_uint num_mip_levels;
	//	cl_uint num_samples;
	//	cl_mem buffer;
	//} cl_image_desc;
	cl_image_desc desc;
	desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	desc.image_width = imageWidth;
	desc.image_height = imageHeight;
	desc.image_depth = 0;
	desc.image_array_size = 0;
	desc.image_row_pitch = 0;
	desc.image_slice_pitch = 0;
	desc.num_mip_levels = 0;
	desc.num_samples = 0;
	desc.buffer = NULL;

	//https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/cl_image_format.html
	//typedef struct _cl_image_format {
        //  cl_channel_order image_channel_order;
        //  cl_channel_type image_channel_data_type;
	//} cl_image_format;
	cl_image_format format;
	format.image_channel_order = CL_R;
	format.image_channel_data_type = CL_FLOAT;

	//https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clCreateImage.html
	//cl_mem clCreateImage (	cl_context context,
	// 	cl_mem_flags flags,
	// 	const cl_image_format *image_format,
	// 	const cl_image_desc *image_desc,
	// 	void *host_ptr,
	// 	cl_int *errcode_ret)
	cl_mem d_inputImage = clCreateImage(context, CL_MEM_READ_ONLY, &format, &desc, NULL, &status);
	cl_mem d_outputImage = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &desc, NULL, &status);

	// 卷积核缓存
	cl_mem d_filter = clCreateBuffer(context, 0, filterSize*sizeof(float), NULL, &status);

	// 主机数据写入设备
	size_t origin[3] = { 0, 0, 0 };
	size_t region[3] = { imageWidth, imageHeight, 1 };// 偏移量和每个维度上的尺寸
	clEnqueueWriteBuffer(commandQueue, d_filter, CL_TRUE, 0, filterSize * sizeof(float), filter, 0, NULL, NULL);

	// 创建采样器，规定图像坐标系的类型和访问越界时的解决方案，以及插值方式
	//https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clCreateSampler.html,
	//cl_sampler clCreateSampler (	cl_context context,
	// 	cl_bool normalized_coords,//图像坐标是否要归一化
	// 	cl_addressing_mode addressing_mode,//图像越界时的处理方案
	// 	cl_filter_mode filter_mode,//Specifies the type of filter that must be applied when reading an image. This can be CL_FILTER_NEAREST or CL_FILTER_LINEAR.即访问值落在多个坐标之间的时候，取最近值或者用差值法
	// 	cl_int *errcode_ret)
	cl_sampler sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &status);


	const char * filename = "rotate.cl";
	std::string sourceStr;
	status = convertToString(filename, sourceStr);
	if (status)
		cout << status << "  !!!!!!!!" << endl;
	const char * source = sourceStr.c_str();
	size_t sourceSize[] = { strlen(source) };

	cl_program program = clCreateProgramWithSource(context, 1, &source, sourceSize, &status);

	status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);
	if (status)//检查编译情况
		cout << status << "  !!!!!!!!" <<endl;
	if (status != 0)
	{
		printf("clBuild failed:%d\n", status);
		char tbuf[0x10000];
		clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0x10000, tbuf,
				NULL);
		printf("\n%s\n", tbuf);
	}

	cl_kernel kernel = NULL;
	kernel = clCreateKernel(program, "rotate_image", &status);

	// 声明内核参数
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_inputImage);
	if (status) cout << "参数设置错误0" << endl;
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_outputImage);
	if (status) cout << "参数设置错误1" << endl;
	status = clSetKernelArg(kernel, 2, sizeof(int), &imageHeight);
	if (status) cout << "参数设置错误2" << endl;
	status = clSetKernelArg(kernel, 3, sizeof(int), &imageWidth);
	if (status) cout << "参数设置错误3" << endl;
	status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_filter);
	if (status) cout << "参数设置错误4" << endl;
	status = clSetKernelArg(kernel, 5, sizeof(int), &filterWidth);
	if (status) cout << "参数设置错误5" << endl;
	status = clSetKernelArg(kernel, 6, sizeof(cl_sampler), &sampler);
	if (status) cout << "参数设置错误6" << endl;

	// 内核参数
	size_t globalSize[2] = { imageWidth, imageHeight };

	int i, j;
	cl_event prof_event;
	cl_ulong ev_start_time = (cl_ulong)0;
	cl_ulong ev_end_time = (cl_ulong)0;
	double rum_time;
	for (i = 0; i < 3; i++)// 三个通道，分别为蓝、绿、红       
	{
		// 更新输入缓冲区
		for (j = 0; j < imageHeight * imageWidth; j++)
			imageData[j] = (float)channel[i].data[j];        

		//https://www.khronos.org/registry/OpenCL/sdk/1.1/docs/man/xhtml/clEnqueueWriteImage.html
		//cl_int clEnqueueWriteImage (	cl_command_queue command_queue,
		// 	cl_mem image,
		// 	cl_bool blocking_write,
		// 	const size_t origin[3],  //Defines the (x, y, z) offset in pixels in the image from where to write. If image is a 2D image object, the z value given by origin[2] must be 0.
		// 	const size_t region[3],  //Defines the (width, height, depth) in pixels of the 2D or 3D rectangle being written. If image is a 2D image object, the depth value given by region[2] must be 1.
		// 	size_t input_row_pitch,
		// 	size_t input_slice_pitch,
		// 	const void * ptr,  //The pointer to a buffer in host memory where image data is to be read from
		// 	cl_uint num_events_in_wait_list,
		// 	const cl_event *event_wait_list,
		// 	cl_event *event)
		clEnqueueWriteImage(commandQueue, d_inputImage, CL_TRUE, origin, region, 0, 0, imageData, 0, NULL, NULL); 

		// 执行内核
		clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalSize, NULL, 0, NULL, &prof_event);

		// 向文件中写入结果
		//https://www.khronos.org/registry/OpenCL/sdk/1.1/docs/man/xhtml/clEnqueueReadImage.html
		//与clEnqueueWriteImage类似，不同的在于
		//	void *ptr, //The pointer to a buffer in host memory where image data is to be written to.
		clEnqueueReadImage(commandQueue, d_outputImage, CL_TRUE, origin, region, 0, 0, imageData, 0, NULL, NULL);
		for (j = 0; j < imageHeight * imageWidth; j++)
			channel[i].data[j] = (imageData[j] < 0 ? 0 : (unsigned char)int(imageData[j]));
	}

	merge(channel, 3, image);                                          // 三个通道合成
	imwrite("./bb_out.jpg", image, vector<int>{IMWRITE_JPEG_QUALITY, 95});// 最后一个参数为输出图片的选项，95%质量
	imshow("merge", image);                                            // 在窗口中展示图片
	waitKey(0);                                                        // 等待键盘输入

	status = clGetEventProfilingInfo(prof_event,CL_PROFILING_COMMAND_QUEUED,
			sizeof(cl_ulong),&ev_start_time,NULL);
	status = clGetEventProfilingInfo(prof_event,CL_PROFILING_COMMAND_END,
			sizeof(cl_ulong),&ev_end_time,NULL);
	if (status) perror("读取时间的时候发生错误\n");
	rum_time = (double)(ev_end_time - ev_start_time);
	cout << "执行时间为:" << rum_time << endl;

	//step12: 释放OpenCL资源和主机内存
	//删除 OpenCL 资源对象
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(commandQueue);
	clReleaseMemObject(d_inputImage);
	clReleaseMemObject(d_outputImage);
	clReleaseContext(context);

	//释放主机内存
	free(imageData);
	free(platforms);
	free(devices);

	return 0;
}
