#include<stdio.h>
#include<stdlib.h>
#include<algorithm>
#include<time.h>
#include<iostream>

using namespace std;

void cpu_avg_pooling(float* cpu_input, float* cpu_output_data, int input_h_size, int input_w_size, int pool_h_size, int pool_w_size, int pool_h_stride, int pool_w_stride)
{
        int sum;
        float avg;

        int pooled_h = ((input_h_size - pool_h_size) / pool_h_stride) + 1;
        int pooled_w = ((input_w_size - pool_w_size) / pool_w_stride) + 1;

        for (int ph = 0; ph < pooled_h; ph++)
        {
                for (int pw = 0; pw < pooled_w; pw++)
                {
                        int h_start = ph * pool_h_stride;
                        int w_start = pw * pool_w_stride;
                        int h_end = min(h_start + pool_h_size, input_h_size);
                        int w_end = min(w_start + pool_w_size, input_w_size);
                        h_start = max(h_start, 0);
                        w_start = max(w_start, 0);
                        sum = 0;
                        avg = 0;
                        int pool_index = (ph * pooled_w) + pw;
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
        int input_h_size = 8;
	int input_w_size = 8;
	int pool_w_size = 7;
        int pool_h_size = 7;
        int pool_w_stride = 1;
        int pool_h_stride = 1;
	
	int pooled_h = ((input_h_size - pool_h_size) / pool_h_stride) + 1;
        int pooled_w = ((input_w_size - pool_w_size) / pool_w_stride) + 1;	

	float* cpu_input = new float[sizeof(float) * input_h_size * input_w_size];
	float* cpu_output_data = new float[sizeof(float) * input_h_size * input_w_size];
	
	Init_input(cpu_input, input_h_size, input_w_size, 10);

	print(cpu_input, input_h_size, input_w_size);

	cpu_avg_pooling(cpu_input, cpu_output_data, input_h_size, input_w_size, pool_h_size, pool_w_size, pool_h_stride, pool_w_stride);

	print(cpu_output_data, pooled_h, pooled_w);
	
	free(cpu_input);
	free(cpu_output_data);

	return 0; 
}




































