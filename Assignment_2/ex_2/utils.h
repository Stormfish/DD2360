#pragma once
#include <sys/time.h>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cuda_runtime.h>
#include "helper_math.h"
#include "utils.h"

double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

std::vector<float> gen_random(int num){
	std::vector<float> r(num);
	srand(0); 
	for (int i = 0; i < num; ++i)
	{
		r[i] = rand(); 
	}
	return r; 
}

float r_bound(float max){
	float r = rand() / (static_cast<float>(RAND_MAX/max));
	return r; 
}

std::vector<float> rand_vector_f(int size){
	//std::srand(unsigned(std::time(nullptr)));
	std::srand(0);
	std::vector<float> v(size);
	std::generate(v.begin(), v.end(), std::rand);
	return v; 
}

std::vector<float3> rand_vector_f3(int size){
	std::srand(0); 
	std::vector<float3> v(size);
	std::generate(v.begin(), v.end(), [](){return make_float3(std::rand(), std::rand(), std::rand());});
	return v; 
}