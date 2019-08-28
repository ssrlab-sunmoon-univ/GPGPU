//offset 부분이랑 width, height 부분 바꿔주기

#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include<algorithm>
#include<time.h>
#include<cuda.h>
#include<mpi.h>

using namespace std;

MPI_Status status;

__global__ void avg_pooling(float* gpu_input, float* gpu_output_data, int input_h_size, int input_w_size, int pool_h_size, int pool_w_size, int pool_h_stride, int pool_w_stride, int width, int before_height, int height) 
{
    int x = blockIdx.x;
    int y = blockIdx.y;

    if((x < width) && (before_height <= y) && (y < height))
    {
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
                sum += gpu_input[index];
            }
            avg = (float)sum / (pool_h_size * pool_w_size);
            gpu_output_data[pool_index] = avg;
        }
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
int main(int argc, char** argv)
{
    int procs, myrank;
    int offset;
    int before_offset = 0;

	int input_h_size = 1024;
	int input_w_size = 1024;
	int pool_w_size = 2;
    int pool_h_size = 2;
    int pool_w_stride = 12;
    int pool_h_stride = 12;
	
	int pooled_h = ((input_h_size - pool_h_size) / pool_h_stride) + 1;
    int pooled_w = ((input_w_size - pool_w_size) / pool_w_stride) + 1;	

	float* input = (float*)malloc(sizeof(float) * input_h_size * input_w_size);
    float* result = (float*)malloc(sizeof(float) * pooled_h * pooled_w);
    float* slave_input = (float*)malloc(sizeof(float) * input_h_size * input_w_size);
    float* slave_result = (float*)malloc(sizeof(float) * pooled_h * pooled_w);
	float* gpu_output_data;
    float* gpu_input;

    clock_t start, end;
    float result_time = 0;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    Init_input(input, input_h_size, input_w_size, 10);

    start = clock();

    // if(myrank == 0)
    // {
    //     print(input, input_h_size, input_w_size);
    // }

	cudaMalloc((void**)&gpu_input, sizeof(float) * input_h_size * input_w_size);
    cudaMalloc((void**)&gpu_output_data, sizeof(float) * pooled_h * pooled_w);
    
    if(myrank == 0)
    {
        int width = pooled_w;
        int height = (pooled_h / procs) * (myrank + 1);
        int before_height = (pooled_h / procs) * myrank;

        for(int i = 1; i < procs; i++)
        {
            MPI_Send(input, input_h_size * input_w_size, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
        }
        cudaMemcpy(gpu_input, input, sizeof(float) * input_h_size * input_w_size, cudaMemcpyHostToDevice);

        dim3 dimGrid(pooled_h, pooled_w);
	    avg_pooling<<<dimGrid,1>>>(gpu_input, gpu_output_data, input_h_size, input_w_size, pool_h_size, pool_w_size, pool_h_stride, pool_w_stride, width, before_height, height);
        cudaDeviceSynchronize();

        cudaMemcpy(result, gpu_output_data, sizeof(float) * pooled_h * pooled_w, cudaMemcpyDeviceToHost);

        offset = (pooled_h / procs);
        for(int i = 1; i < procs; i++)
        {
            MPI_Recv(slave_result, pooled_h * pooled_w, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &status);
            before_offset = offset;
            offset += (pooled_h / procs);
            for(int h = before_offset; h < offset; h++)
            {
                for(int w = 0; w < pooled_h; w++)
                {
                    result[(h * pooled_h + w)] = result[(h * pooled_h + w)] + slave_result[(h * pooled_h + w)];
                }
            }
        }
    }
    else if(myrank > 0)
    {
        int width = pooled_w;
        int height = (pooled_h / procs) * (myrank + 1);
        int before_height = (pooled_h / procs) * myrank;

        MPI_Recv(slave_input, input_h_size * input_w_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);

        cudaMemcpy(gpu_input, slave_input, sizeof(float) * input_h_size * input_w_size, cudaMemcpyHostToDevice);

        dim3 dimGrid(pooled_h, pooled_w);
        avg_pooling<<<dimGrid,1>>>(gpu_input, gpu_output_data, input_h_size, input_w_size, pool_h_size, pool_w_size, pool_h_stride, pool_w_stride, width, before_height, height);
        cudaDeviceSynchronize();

        cudaMemcpy(slave_result, gpu_output_data, sizeof(float) * pooled_h * pooled_w, cudaMemcpyDeviceToHost);

        MPI_Send(slave_result, pooled_h * pooled_w, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
    }
    cudaDeviceSynchronize();
    end = clock();

    result_time = (float)(end-start)/CLOCKS_PER_SEC;

    // if(myrank == 0)
    // {
    //     print(result, pooled_h, pooled_w);                                                      
    // }

    printf("rank = %d time %.4f\n", myrank,result_time);
    
    free(input);
    free(result);
    free(slave_input);
    free(slave_result);
    
	cudaFree(gpu_output_data);
    cudaFree(gpu_input);

    MPI_Finalize();
	return 0; 
}