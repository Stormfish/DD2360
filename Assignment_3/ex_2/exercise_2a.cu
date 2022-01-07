#include <stdio.h>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <math.h>

#include "utils.h"

constexpr int num_iterations = 10; 
constexpr int num_particles = 100000000; 
constexpr int num_threads = 256; 
constexpr int num_blocks = num_particles/num_threads + 1; 
__device__ float* randoms; 


struct Particle{
	float3 pos; 
	float3 vel; 
};

__device__
void vel_update(Particle* p, Particle c){ 
    const unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;  
    if(tid >= num_particles){ 
    	return; 
    }
}

__device__ 
void pos_update(Particle* p, float dt){ 
    const unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;  
    if(tid >= num_particles){ 
    	return; 
    }
    p[tid].pos+=p[tid].vel*dt; 
}

__global__
void update(Particle* p, float dt){
	const unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;  
    if(tid >= num_particles){ 
    	return; 
    } 

    //p-update
	p[tid].pos+=p[tid].vel*dt; 

	//v-update, 1% slowdown per dt
	p[tid].vel=p[tid].vel*dt*0.99; 
} 

void cpu_update(Particle* p, float dt){

	for(int i=0; i<num_particles; ++i){
	    //p-update
		p[i].pos+=p[i].vel*dt; 

		//v-update, 1% slowdown per dt
		p[i].vel-=p[i].vel*dt*0.99; 
	} 
}


int main(){
	//generate random velocities & positions
	Particle* pm; 
	cudaHostAlloc(&pm, sizeof(Particle)*num_particles, cudaHostAllocDefault);

	srand(0); 
	for(int i=0; i<num_particles; ++i){

		pm[i].vel = {r_bound(1000),r_bound(1000),r_bound(1000)}; 
		pm[i].pos = {r_bound(1000),r_bound(1000),r_bound(1000)}; 
	}



	Particle* d_pv;
	cudaMalloc( (void **)&d_pv, num_particles*sizeof(Particle) ); 
	double iStart = 0;
	double iElaps = 0;
	iStart = cpuSecond();

	for(int i=0; i<num_iterations; ++i){
		cudaMemcpy( (void*)d_pv, pm, num_particles*sizeof(Particle), cudaMemcpyHostToDevice ); 
		update<<<num_blocks, num_threads>>>(d_pv, 1.0f); 
		cudaMemcpy(pm, (void*)d_pv, num_particles*sizeof(Particle), cudaMemcpyDeviceToHost ); 
	}
	iElaps = cpuSecond() - iStart;

	printf("Done! in:%fs \n", iElaps); 

	
	return 0; 
}