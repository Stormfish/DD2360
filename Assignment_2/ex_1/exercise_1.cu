#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

__global__ void print_kernel(){
    const unsigned int tid = threadIdx.x;
	printf("printed from: %u\n", tid); 

}

int main(){
	int grid = 1;
	int threads = 256; 
	printf("launch kernels:\n");
    print_kernel<<< grid,threads >>>();
    cudaDeviceSynchronize();
    printf("\nkernels done\n");
	
	return 0; 
}

