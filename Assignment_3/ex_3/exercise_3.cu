#include <stdio.h>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <math.h>

#include "utils.h"

constexpr int num_iterations = 10; 
constexpr int num_particles = 10000000; 
constexpr int num_threads = 128; 
constexpr int num_blocks = num_particles/num_threads + 1; 
constexpr int num_streams = 4; 

__device__ float* randoms; 

struct Particle{
	float3 pos; 
	float3 vel; 
};


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
		p[i].vel=p[i].vel*dt*0.99; 
	} 
}


int main(){
	//generate random velocities & positions
	Particle* pm; 
	cudaMallocHost(&pm, num_particles*sizeof(Particle), cudaHostAllocDefault);

	cudaStream_t streams[num_streams]; 
	for(int i=0; i<num_streams; ++i){
		cudaStreamCreate(&streams[i]); 
	}

	srand(0); 
	for(int i=0; i<num_particles; ++i){
		pm[i].vel = {r_bound(1000),r_bound(1000),r_bound(1000)}; 
		pm[i].pos = {r_bound(1000),r_bound(1000),r_bound(1000)}; 
	}


	Particle* d_pv;
	cudaMalloc( (void **)&d_pv, num_particles*sizeof(Particle) ); 
	
	double iStart = 0;
	double iElaps = 0;

	float sum=0; 
	for(int i=0; i<num_particles; ++i){
		sum+=pm[i].vel.x; 
	}
	printf("Starting! sum is:%f \n", sum); 


	iStart = cpuSecond();
	for(int i=0; i<num_iterations; ++i){
		for(int s=0; s<num_streams; ++s)
			cudaMemcpyAsync( (void*)d_pv, pm, num_particles*sizeof(Particle), cudaMemcpyHostToDevice, streams[s] ); 
		for(int s=0; s<num_streams; ++s)
			update<<<num_blocks, num_threads, 0, streams[s]>>>(d_pv, 1.0f); 
		for(int s=0; s<num_streams; ++s)
			cudaMemcpyAsync(pm, (void*)d_pv, num_particles*sizeof(Particle), cudaMemcpyDeviceToHost, streams[s] ); 
			cudaDeviceSynchronize();
		
		sum=0; 
		for(int p=0; p<num_particles; ++p){
			sum+=pm[p].vel.x; 
		}
		printf("sum:%d\t was:%f \n",i, sum); 
	}
	iElaps = cpuSecond() - iStart;

	for(int i=0; i<num_streams; ++i){
		cudaStreamDestroy(streams[i]); 
	}

	sum=0; 
	for(int i=0; i<num_particles; ++i){
		sum+=pm[i].vel.x; 
	}
	printf("Done! in:%fs, sum was:%f \n", iElaps, sum); 

	
	return 0; 
}