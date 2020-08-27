// convolution.cl，核函数
__kernel void rotate_image(__read_only image2d_t sourceImage, __write_only image2d_t outputImage,
		int rows, int cols, __constant float* filter, int filterWidth, sampler_t sampler)
{
	// 注意工作项的顺序，图像上是先横着数再竖着数
	const int col = get_global_id(0), row = get_global_id(1);
	const int halfWidth = (int)(filterWidth/2);
	
	// 输出数据类型是四元浮点数，与 image 统一
	float4 sum = { 0.0f, 0.0f, 0.0f, 0.0f }, pixel;

	// 传入的卷积窗口是一维的，用一个下标即可遍历
	int i, j, filterIdx;
	int2 coords;
	for (filterIdx = 0, i = -halfWidth; i <= halfWidth; i++)
	{
		// 从 work-item 分到的行号偏移 i 行，作为图像坐标的第二分量
		coords.y = row + i; 
		for (j = -halfWidth; j <= halfWidth; j++)
		{
			// 从 work-item 分到的列号偏移 i 列，作为图像坐标的第一分量
			coords.x = col + j; 
			pixel = read_imagef(sourceImage, sampler, coords);  // 读取源图像上相应位置的值
			sum.x += pixel.x * filter[filterIdx++];
		}
	}
	if (row < rows && col < cols)                   // 将落在有效范围内的计算数据输出
	{
		coords.x = col;
		coords.y = row;
		write_imagef(outputImage, coords, sum);
	}
	return;
}
