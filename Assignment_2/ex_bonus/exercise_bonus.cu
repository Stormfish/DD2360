#include <stdio.h>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include <math.h>
#include "utils.h"

constexpr int NUM_ITER = 1000;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int BLOCKS_PER_KERNEL = 256;
constexpr int TOTAL_THREADS		= THREADS_PER_BLOCK*BLOCKS_PER_KERNEL; 
constexpr int TOTAL_ITERATIONS  = THREADS_PER_BLOCK*BLOCKS_PER_KERNEL*NUM_ITER; 

struct Particle{
	float3 pos; 
	float3 vel; 
};

__global__ void gpu_random(curandState *states, float* res) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid >= TOTAL_THREADS){ 
    	return; 
    } 

    float x;
    float y; 
    int seed = 0+tid; 

	curand_init(seed, tid, 0, &states[tid]);  // 	Initialize CURAND
	for(int i = 0; i < NUM_ITER; i++) {
		x = curand_uniform (&states[tid]);
		y = curand_uniform (&states[tid]);
		//x²+y²=<r²?
		if(x*x+y*y<=1.0){
			res[tid]+=1.0; 
		} 
		//printf("tid:%d %f,%f, %f\n",tid, x, y, res[tid]);
	}	

}


int main(){
	//generate random velocities & positions
	curandState *dev_random;
	cudaMalloc((void**)&dev_random, TOTAL_THREADS*sizeof(curandState));
	float* d_hits;
	cudaMalloc((void**)&d_hits, 	TOTAL_THREADS*sizeof(curandState));

	double iStart = cpuSecond();
	gpu_random<<<BLOCKS_PER_KERNEL, THREADS_PER_BLOCK>>>(dev_random, d_hits); 
	cudaDeviceSynchronize();



	std::vector<float> results(TOTAL_THREADS);
	cudaMemcpy(results.data(), (void*)d_hits, results.size()*sizeof(float), cudaMemcpyDeviceToHost); 
	double iElaps = cpuSecond() - iStart;
	printf("Done! in:%fs \n", iElaps); 
	float sum=0; 
	for (auto& f:results)
	{
		sum += f; 
	}

	float pi = 4*sum/TOTAL_ITERATIONS; 
	printf("sum:%f\n", sum); 
	printf("pi=%f\n", pi);
	return 0; 
}