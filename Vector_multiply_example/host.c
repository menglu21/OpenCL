#include <CL/cl.h>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <fstream>

using namespace std;

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
	//step1: 创建平台对象，平台对象就是指不同供应商提供的主机与计算单元之间联系的实现，比如AMD和NV有不同的平台。通过clGetPlatformIDs函数来获取已有平台。clGetPlatformIDs函数调用两次。
	//clGetPlatformIDs: https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clGetPlatformIDs.html
	//cl_int clGetPlatformIDs(cl_uint num_entries,cl_platform_id *platforms,cl_uint *num_platforms)
	cl_int status;
	cl_uint numPlatforms=0;
	status = clGetPlatformIDs(0, NULL, &numPlatforms);//第一次调用
	//cout << "numPlatforms:" << numPlatforms << endl;//查看有几种平台
	cl_platform_id* platforms = new cl_platform_id[numPlatforms];//分配空间给所有平台
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);//第二次调用，将平台传递给相应的指针

	//查看所有平台的信息
	//char pform_name[40];
	//for (int i=0; i<numPlatforms; i++)
	//{
	//	err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(pform_name), &pform_name, NULL);
	//	cout << "platform " << i <<"name: " << pform_name<<endl;

	//	clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(pform_name), &pform_name, NULL);
	//	cout << "platform " << i << "vendor: " << pform_name<<endl;

	//	clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(pform_name), &pform_name, NULL);
	//	cout << "platform " << i << "version: " << pform_name<<endl;

	//	clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, sizeof(pform_name), &pform_name, NULL);
	//	cout << "platform " << i << "profile: " << pform_name<<endl;
	//}

	//step2: 创建GPU 设备,clGetDeviceIDs, 如果存在多个平台，函数中需要指定平台。和创建平台对象类似，也需要几个步骤，发现设备个数，分配空间，将设备与指针对应。
	//https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clGetDeviceIDs.html, 
	//cl_int clGetDeviceIDs(cl_platform_id platform,cl_device_type device_type,cl_uint num_entries, 	cl_device_id *devices, cl_uint *num_devices)
	cl_uint numDevices=0;
	cl_device_id* devices=NULL;
	status=clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);//第一次调用
	devices=(cl_device_id*) malloc(numDevices*sizeof(cl_device_id));
	status=clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);//第二次调用

	//step3: 创建上下文context, clCreateContext函数，
	//https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clCreateContext.html
	//cl_context clCreateContext(cl_context_properties *properties,	cl_uint num_devices, const cl_device_id *devices, void *pfn_notify (const char *errinfo, const void *private_info, size_t cb, void *user_data),	void *user_data, cl_int *errcode_ret)
	cl_context context=NULL;
	context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);

	//step4: 创建命令队列，命令队列是主机端用于向设备端发送请求的行为机制。一旦主机端指定运行kernel的设备，并且上下文已新建，那么每个设备必须新建一个命令队列（即每个命令队列只关联一个设备）。只要主机需要对设备进行操作，就会把命令提交道正确的命令队列。
	//clCreateCommandQueue, https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clCreateCommandQueue.html
	//cl_command_queue clCreateCommandQueue(cl_context context,cl_device_id device,	cl_command_queue_properties properties,	cl_int *errcode_ret)

	cl_command_queue commandQueue;
	commandQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &status);
	if (commandQueue == NULL) perror("Failed to create commandQueue for device 0.");

	// 建立要传入设备的数据，此例为矩阵乘法
	const int Ndim = 4;
	const int Mdim = 5;
	const int Pdim = 6;
	int szA = Ndim * Pdim;
	int szB = Pdim * Mdim;
	int szC = Ndim * Mdim;

	float *A;
	float *B;
	float *C;

	A = (float *)malloc(szA * sizeof(float));
	B = (float *)malloc(szB * sizeof(float));
	C = (float *)malloc(szC * sizeof(float));
	int i, j;
	for (i = 0; i < szA; i++)
		A[i] = (float)((float)i + 1.0);
	for (i = 0; i < szB; i++)
		B[i] = (float)((float)i + 1.0);

	//step5: 创建buffer对象, clCreateBuffer, https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clCreateBuffer.html
	//cl_mem clCreateBuffer(cl_context context,cl_mem_flags flags,size_t size,void *host_ptr,cl_int *errcode_ret)

	cl_mem memObjects[3] = { 0, 0, 0 };
	memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY |  CL_MEM_COPY_HOST_PTR,
			sizeof(float)* szA, A, &status);//此处第四个变量用A的话可以直接将A写入创建的设备buffer中
	memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY |  CL_MEM_COPY_HOST_PTR,
			sizeof(float)* szB, B, &status);
	memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			sizeof(float)* szC, C, &status);//由于数组C为输出值，所有格式为"READ_WRITE"
	if (memObjects[0] == NULL || memObjects[1] == NULL ||memObjects[2] == NULL)
		perror("Error in clCreateBuffer.\n");

	//导入kernel函数
	const char * filename = "Vadd.cl";
	std::string sourceStr;
	status = convertToString(filename, sourceStr);
	if (status)
		cout << status << "  !!!!!!!!" << endl;
	const char * source = sourceStr.c_str();
	size_t sourceSize[] = { strlen(source) };

	//step6: 创建程序对象clCreateProgramWithSource, https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clCreateProgramWithSource.html
	//cl_program clCreateProgramWithSource (cl_context context, cl_uint count, const char **strings, const size_t *lengths,	cl_int *errcode_ret)
	cl_program program = clCreateProgramWithSource(context, 1, &source, sourceSize, &status);

	//step7: 编译程序对象clBuildProgram, https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clBuildProgram.html
	//cl_int clBuildProgram (cl_program program, cl_uint num_devices, const cl_device_id *device_list, const char *options, void (*pfn_notify)(cl_program, void *user_data), void *user_data)
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

	//step8: 创建Kernel对象，https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clCreateKernel.html
	//cl_kernel clCreateKernel(cl_program  program,	const char *kernel_name, cl_int *errcode_ret)
	cl_kernel kernel = NULL;
	kernel = clCreateKernel(program, "matrix_mult", &status);//第二个参数要和Kernel函数一致

	//step9: 设置 Kernel 参数,https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clSetKernelArg.html
	//cl_int clSetKernelArg( cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value)
	//此例有六个参数，三个表示矩阵维度，三个为矩阵的buffer
	status = clSetKernelArg(kernel, 0, sizeof(int), &Ndim);
	status = clSetKernelArg(kernel, 1, sizeof(int), &Mdim);
	status = clSetKernelArg(kernel, 2, sizeof(int), &Pdim);
	status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &memObjects[0]);
	status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &memObjects[1]);
	status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &memObjects[2]);
	if (status) cout << "参数设置错误" << endl;

	//step10:配置work-item的结构，并执行。比如此例中，输出的为二维矩阵，因此可以让输出矩阵每一个元素的计算都分配给一个kernel。
	size_t global[2];//所有work-item的索引坐标，此例中为二维
	cl_event prof_event;
	cl_ulong ev_start_time = (cl_ulong)0;
	cl_ulong ev_end_time = (cl_ulong)0;
	double rum_time;
	global[0] = (size_t)Ndim;
	global[1] = (size_t)Mdim;
	//clEnqueueNDRangeKernel，https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clEnqueueNDRangeKernel.html
	//cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue,
	// 			 	cl_kernel kernel,
	// 			 	cl_uint work_dim,
	// 			 	const size_t *global_work_offset,
	// 			 	const size_t *global_work_size,
	// 			 	const size_t *local_work_size,
	// 			 	cl_uint num_events_in_wait_list,
	// 			 	const cl_event *event_wait_list,
	// 			 	cl_event *event)
	status = clEnqueueNDRangeKernel(commandQueue, kernel,2,NULL,global, NULL, 0, NULL, &prof_event);//变量中的2对应global的数组长度，为索引坐标的维度
	if (status)
		cout << "执行内核时错误" << endl;
	clFinish(commandQueue);

	//额外的信息：读取时间
	status = clGetEventProfilingInfo(prof_event,CL_PROFILING_COMMAND_QUEUED,
			sizeof(cl_ulong),&ev_start_time,NULL);
	status = clGetEventProfilingInfo(prof_event,CL_PROFILING_COMMAND_END,
			sizeof(cl_ulong),&ev_end_time,NULL);
	if (status) perror("读取时间的时候发生错误\n");
	rum_time = (double)(ev_end_time - ev_start_time);
	cout << "执行时间为:" << rum_time << endl;

	//step11: 读取输出，并copy回主机内存clEnqueueReadBuffer，https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clEnqueueReadBuffer.html
	//cl_int clEnqueueReadBuffer (	cl_command_queue command_queue,
	// 			 	cl_mem buffer,//输出对应的buffer
	// 			 	cl_bool blocking_read,
	// 			 	size_t offset,
	// 			 	size_t cb,
	// 			 	void *ptr,
	// 			 	cl_uint num_events_in_wait_list,
	// 			 	const cl_event *event_wait_list,
	// 			 	cl_event *event)
	status = clEnqueueReadBuffer(commandQueue, memObjects[2],CL_TRUE, 0,
			sizeof(float)* szC, C,0, NULL, NULL);
	if (status)
		perror("读回数据的时候发生错误\n");

	//额外信息：结果显示
	printf("\nArray A:\n");
	for (i = 0; i < Ndim; i++) {
		for (j = 0; j < Pdim; j++)
			printf("%.3f\t", A[i*Pdim + j]);
		printf("\n");
	}
	printf("\nArray B:\n");
	for (i = 0; i < Pdim; i++) {
		for (j = 0; j < Mdim; j++)
			printf("%.3f\t", B[i*Mdim + j]);
		printf("\n");
	}
	printf("\nArray C:\n");
	for (i = 0; i < Ndim; i++) {
		for (j = 0; j < Mdim; j++)
			printf("%.3f\t", C[i*Mdim + j]);
		printf("\n");
	}
	cout << endl;

	//step12: 释放OpenCL资源和主机内存
	//删除 OpenCL 资源对象
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(commandQueue);
	clReleaseMemObject(memObjects[0]);
	clReleaseMemObject(memObjects[1]);
	clReleaseMemObject(memObjects[2]);
	clReleaseContext(context);

	//释放主机内存
	if (A)
		free(A);
	if (B)
		free(B);
	if (C)
		free(C);
	free(platforms);
	free(devices);

	return 0;
}
