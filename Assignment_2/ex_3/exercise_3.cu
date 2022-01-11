#include <stdio.h>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <math.h>
#include <getopt.h>

#include "utils.h"

constexpr int num_iterations = 100; 
int num_particles = 100000; 
int num_threads = 256; 
int num_blocks = num_particles/num_threads + 1; 
__device__ float* randoms; 


struct Particle{
	float3 pos; 
	float3 vel; 
};


__global__
void update(Particle* p, float dt, int num_particles){
	const unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;  
    if(tid >= num_particles){ 
    	return; 
    } 

    //p-update
	p[tid].pos+=p[tid].vel*dt; 

	//v-update, 50% slowdown per dt
	p[tid].vel=p[tid].vel*dt*0.50; 
} 

void cpu_update(Particle* p, float dt){

	for(int i=0; i<num_particles; ++i){
	    //p-update
		p[i].pos+=p[i].vel*dt; 

		//v-update, 50% slowdown per dt
		p[i].vel=p[i].vel*dt*0.50; 
	} 
}


int main(int argc, char *argv[]){
	int opt;
	while ((opt = getopt(argc, argv, "s::b::")) != -1) {
		switch (opt) {
			case 's':
				num_particles = atoi(optarg);
				break; 
			case 'b':
				num_threads = atoi(optarg);
				break;
			default:
				printf("invalid options, exiting"); 
				exit(1);
		}
	}

	num_blocks  = num_particles/num_threads + 1;
	//printf("particles:%d, blocks:%d, threads:%d\n", num_particles, num_blocks, num_threads); 
	//fprintf( stderr, "particles:%d, blocks:%d, threads:%d\n", num_particles, num_blocks, num_threads);

	//generate random velocities & positions
	std::vector<Particle> pv(num_particles);
	srand(0); 
	for(auto& p:pv){
		p.vel = {r_bound(1000),r_bound(1000),r_bound(1000)}; 
		p.pos = {r_bound(1000),r_bound(1000),r_bound(1000)}; 
	}

	Particle* d_pv;
	cudaMalloc( (void **)&d_pv, num_particles*sizeof(Particle) ); 
	double iStart = cpuSecond();
	cudaMemcpy( (void*)d_pv, pv.data(), pv.size()*sizeof(Particle), cudaMemcpyHostToDevice ); 
	for(int i=0; i<num_iterations; ++i){
		update<<<num_blocks, num_threads>>>(d_pv, 1.0f, num_particles); 
			cudaDeviceSynchronize();

	}
	cudaMemcpy(pv.data(), (void*)d_pv, pv.size()*sizeof(Particle), cudaMemcpyDeviceToHost ); 
	cudaDeviceSynchronize();
	double gpuElaps = cpuSecond() - iStart;

	//printf("Done! in:%fs \n", gpuElaps); 

	//CPU
	std::vector<Particle> pvc(num_particles);
	srand(0);
	for(auto& p:pvc){
		p.vel = {r_bound(1000),r_bound(1000),r_bound(1000)}; 
		p.pos = {r_bound(1000),r_bound(1000),r_bound(1000)}; 
	}
	iStart = cpuSecond();
	for(int i=0; i<num_iterations; ++i){
		cpu_update(pvc.data(), 1.0f); 
	}
	double cpuElaps = cpuSecond() - iStart;
	//printf("Done! in:%fs \n", cpuElaps); 

	//Compare, only pos x is fine
	float errsum = 0; 
	for(int i=0; i<num_particles; ++i){
		errsum+=pvc[i].pos.x-pv[i].pos.x; 
	}
	//printf("sum of errors: %f\n", errsum); 
	//printf("gpu(s)\t cpu(s)\n %f\t%f\n", gpuElaps, cpuElaps);
	printf("%f\t%f\n", gpuElaps, cpuElaps); 


	return 0; 
}