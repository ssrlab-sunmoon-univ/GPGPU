#include<stdio.h>
#include<stdlib.h>
#include<algorithm>
#include<time.h>
#include<iostream>
#include<mpi.h>

using namespace std;

MPI_Status status;

void cpu_avg_pooling(float* cpu_input, float* cpu_output_data, int input_h_size, int input_w_size, int pool_h_size, int pool_w_size, int pool_h_stride, int pool_w_stride, int width, int before_height, int height)
{
        int sum;
        float avg;

        int pooled_h = ((input_h_size - pool_h_size) / pool_h_stride) + 1;
        int pooled_w = ((input_w_size - pool_w_size) / pool_w_stride) + 1;
        
        for(int y = 0; y < pooled_h; y++)
        {
                for(int x = 0; x < pooled_w; x++)
                {
                        if((x < width) && (before_height <= y) && (y < height))
                        {
                                int h_start = y * pool_h_stride;
                                int w_start = x * pool_w_stride;
                                int h_end = min(h_start + pool_h_size, input_h_size);
                                int w_end = min(w_start + pool_w_size, input_w_size);

                                h_start = max(h_start, 0);
                                w_start = max(w_start, 0);

                                sum = 0;
                                avg = 0;
                                int pool_index = (y * pooled_w) + x;
                                for (int h = h_start; h < h_end; h++)
                                {
                                        for (int w = w_start; w < w_end; w++)
                                        {
                                                int index = (h * input_w_size) + w;
                                                sum += cpu_input[index];
                                        }
                                }
                                avg = (float)sum / (pool_h_size * pool_w_size);
                                cpu_output_data[pool_index] = avg;  
                        }
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
        int before_offset;
        float sum = 0;
        float avg = 0;

        clock_t start, end;
        float result_time = 0;
        
        int input_h_size = 2304;
	int input_w_size = 2304;
	int pool_w_size = 4;
        int pool_h_size = 4;
        int pool_w_stride = 1;
        int pool_h_stride = 1;
	
	int pooled_h = ((input_h_size - pool_h_size) / pool_h_stride) + 1;
        int pooled_w = ((input_w_size - pool_w_size) / pool_w_stride) + 1;	

	float* cpu_input = new float[sizeof(float) * input_h_size * input_w_size];
	float* cpu_output_data = new float[sizeof(float) * pooled_h * pooled_w];
        float* slave_input = new float[sizeof(float) * input_h_size * input_w_size];
        float* slave_output_data = new float[sizeof(float) * pooled_h * pooled_w];

        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &procs);
	
	Init_input(cpu_input, input_h_size, input_w_size, 10);

	for(int i = 0; i < 50; i++){
                result_time = 0;
                start = 0;
                end = 0;
                if(myrank == 0)
                {
                        int width = pooled_w;
                        int height = (pooled_h / procs) * (myrank + 1);
                        int before_height = (pooled_h / procs) * myrank;

                        for(int i = 1; i < procs; i++)
                        {
                                MPI_Send(cpu_input, input_h_size * input_w_size, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                        }

                        start = clock();

	                cpu_avg_pooling(cpu_input, cpu_output_data, input_h_size, input_w_size, pool_h_size, pool_w_size, pool_h_stride, pool_w_stride, width, before_height, height);

                        offset = (pooled_h / procs);

                        for(int i = 1; i < procs; i++)
                        {
                                MPI_Recv(slave_output_data, pooled_h * pooled_w, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &status);
                                before_offset = offset;
                                offset += (pooled_h / procs);
                                for(int h = before_offset; h < offset; h++)
                                {
                                        for(int w = 0; w < pooled_h; w++)
                                        {
                                                cpu_output_data[(h * pooled_h + w)] = cpu_output_data[(h * pooled_h + w)] + slave_output_data[(h * pooled_h + w)];
                                        }
                                }
                        }
                        end = clock();
                }
                else if(myrank > 0)
                {
                        int width = pooled_w;
                        int height = (pooled_h / procs) * (myrank + 1);
                        int before_height = (pooled_h / procs) * myrank;

                        MPI_Recv(slave_input, input_h_size * input_w_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);

                        cpu_avg_pooling(cpu_input, cpu_output_data, input_h_size, input_w_size, pool_h_size, pool_w_size, pool_h_stride, pool_w_stride, width, before_height, height);
                        MPI_Send(slave_output_data, pooled_h * pooled_w, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
      	        }
                result_time = (float)(end - start) / CLOCKS_PER_SEC;

                sum += result_time;
        }

        avg = sum / 50.0;

        if(myrank == 0)
        {
                printf("time => %.4f\n", avg);
        }

	free(cpu_input);
	free(cpu_output_data);
        free(slave_input);
        free(slave_output_data);

        MPI_Finalize();
	return 0; 
}




































