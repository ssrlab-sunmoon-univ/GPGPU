#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include<algorithm>
#include<time.h>
#include<cuda.h>

using namespace std;

__global__ void avg_pooling(float* dev, float* gpu_output_data, int input_h_size, int input_w_size, int pool_h_size, int pool_w_size, int pool_h_stride, int pool_w_stride)
{
        int x = blockIdx.x;
	int y = blockIdx.y;

	int sum;
        float avg;

        int pooled_size = ((input_w_size - pool_w_size) / pool_w_stride) + 1;
	int h_start = y * pool_h_stride;
        int w_start = x * pool_w_stride;
        int h_end = min(h_start + pool_h_size, input_h_size);
        int w_end = min(w_start + pool_w_size, input_w_size);

        h_start = max(h_start, 0);
        w_start = max(w_start, 0);
        sum = 0;
        avg = 0;

        int pool_index = (y * pooled_size) + x;
        for (int h = h_start; h < h_end; h++)
        {
            for (int w = w_start; w < w_end; w++)
            {
                  	int index = (h * input_w_size) + w;
                   	sum += dev[index];
            }
        avg = (float)sum / (pool_h_size * pool_w_size);
        gpu_output_data[pool_index] = avg;
        }
}
void Init_input(float* input, int input_h_size, int input_w_size, int num)
{
        srand(time(NULL));

        for (int h = 0; h < input_h_size; h++)
        {
        	for (int w = 0; w < input_w_size; w++)
                {
                	input[(h * input_w_size) + w] = rand() % num;
                }
        }

}
void print(float* data, int h_size, int w_size)
{
	for (int h = 0; h < h_size; h++)
        {
                for (int w = 0; w < w_size; w++)
                {
           	        printf("%.2f ", data[(h * w_size) + w]);
		}
	        printf("\n");
        }
        printf("\n");
}
int main()
{
	int input_h_size = 100;
	int input_w_size = 100;
	int pool_w_size = 99;
        int pool_h_size = 99;
        int pool_w_stride = 1;
        int pool_h_stride = 1;
	
	int pooled_h = ((input_h_size - pool_h_size) / pool_h_stride) + 1;
        int pooled_w = ((input_w_size - pool_w_size) / pool_w_stride) + 1;	

	float* input = (float*)malloc(sizeof(float) * input_h_size * input_w_size);
	float* result = (float*)malloc(sizeof(float) * input_h_size * input_w_size);
	float* cpu_result = (float*)malloc(sizeof(float) * input_h_size * input_w_size);
	float* gpu_output_data;
	float* dev;

	Init_input(input, input_h_size, input_w_size, 10);

	print(input, input_h_size, input_w_size);

	cudaMalloc((void**)&dev, sizeof(float) * input_h_size * input_w_size);
	cudaMalloc((void**)&gpu_output_data, sizeof(float) * input_h_size * input_w_size);

	cudaMemcpy(dev, input, sizeof(float) * input_h_size * input_w_size, cudaMemcpyHostToDevice);
	
	dim3 dimGrid(pooled_h, pooled_w);
	avg_pooling<<<dimGrid,1>>>(dev, gpu_output_data, input_h_size, input_w_size, pool_h_size, pool_w_size, pool_h_stride, pool_w_stride);
	
	cudaMemcpy(result, gpu_output_data, sizeof(float) * input_h_size * input_w_size, cudaMemcpyDeviceToHost);

	print(result, pooled_h, pooled_w);
	
	cudaFree(gpu_output_data);
	cudaFree(dev);
	free(input);
	free(result);

	return 0; 
}



































