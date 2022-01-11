#include <stdio.h>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#include "utils.h"


constexpr int ARRAY_SIZE = 1000000; 
constexpr int num_threads = 256; 
constexpr int num_blocks = ARRAY_SIZE/num_threads + 1; 

__global__ 
void saxpy(float a, float* x, float* y){ 
    const unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;  
    if(tid >= ARRAY_SIZE){ 
    	return; 
    }
    y[tid] = a*x[tid] + y[tid]; 
}

void saxpy_cpu(float a, float* x, float* y){
    for(int i=0; i<ARRAY_SIZE; ++i){ 
    	y[i] = a*x[i] + y[i]; 
    }
}

int main(){

	std::vector<float> v(ARRAY_SIZE, 5.0); 
	float* x_array; 
	float* y_array; 

	cudaMalloc( (void **)&x_array, v.size()*sizeof(float) ); 
	cudaMalloc( (void **)&y_array, v.size()*sizeof(float) ); 
	cudaMemcpy( (void*)x_array, v.data(), v.size()*sizeof(float), cudaMemcpyHostToDevice ); 
	cudaMemcpy( (void*)y_array, v.data(), v.size()*sizeof(float), cudaMemcpyHostToDevice ); 


	printf("Computing SAXPY on the GPU…:"); 
	double iStart = cpuSecond();
	saxpy<<<num_blocks, num_threads>>>(5, x_array, y_array); 
	cudaDeviceSynchronize();
	double iElaps = cpuSecond() - iStart;
	cudaMemcpy( v.data(), (void*)y_array, v.size()*sizeof(float), cudaMemcpyDeviceToHost );	
	printf("Done! in:%fs \n", iElaps); 

	std::vector<float> v2(ARRAY_SIZE, 5.0); 
	printf("Computing SAXPY on the CPU…:"); 
	iStart = cpuSecond();
	saxpy_cpu(5, v2.data(), v2.data()); 
	iElaps = cpuSecond() - iStart;
	printf("Done! in:%fs \n", iElaps); 

	float diff=0.0; 
	for(int i=0; i<ARRAY_SIZE; ++i){
		diff += v[i]-v2[i]; 
	} 
	printf("\nsum of differences: %f\n", diff); 

	


	return 0; 
}

