#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>


__global__ void print_kernel(){
    const unsigned int tid = threadIdx.x;
	printf("printed from: %u", tid); 
}
