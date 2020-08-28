// convolution.cl，核函数
__kernel void rotate_image(__read_only image2d_t sourceImage, __write_only image2d_t outputImage,
		int rows, int cols, __constant float* filter, int filterWidth, sampler_t sampler)
{
	const int pixel_x = get_global_id(0);//像素横坐标
	const int pixel_y = get_global_id(1);//像素纵坐标
	const int halfWidth = (int)(filterWidth/2);
	
	// 输出数据类型是四元浮点数，与 image 统一
	float4 sum = { 0.0f, 0.0f, 0.0f, 0.0f }, pixel;//四元浮点矢量

	// 传入的卷积窗口是一维的，用一个下标即可遍历
	int i, j, filterIdx;
	int2 coords;
	for (filterIdx = 0, i = -halfWidth; i <= halfWidth; i++)
	{
		coords.y = pixel_y + i; 
		for (j = -halfWidth; j <= halfWidth; j++)
		{
			coords.x = pixel_x + j; 
			pixel = read_imagef(sourceImage, sampler, coords);  // 读取源图像上相应位置的值
			sum.x += pixel.x * filter[filterIdx++];
		}
	}
	if (pixel_y < rows && pixel_x < cols)                   // 将落在有效范围内的计算数据输出
	{
		coords.x = pixel_x;
		coords.y = pixel_y;
		write_imagef(outputImage, coords, sum);
	}
	return;
}
