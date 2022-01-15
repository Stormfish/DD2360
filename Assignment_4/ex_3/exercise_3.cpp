#include <iostream>
#include <vector>
#include <getopt.h>

#include "utils.h"
int size = 4000000; 

using std::vector;


void cpu_saxpy(int size, float a, float* X, float* Y){
    for(int i=0; i<size; ++i){ 
        Y[i] = a*X[i] + Y[i]; 
    }
}

void acc_saxpy(float* res, int size, float a, float* X, float* Y){
    #pragma acc parallel loop copyin(X[0:size]) copyin(Y[0:size]) copyout(res[0:size])
    for(int i=0; i<size; ++i){ 
        res[i] = a*X[i] + Y[i]; 
    }
}




int main(int argc, char *argv[]) {
	int opt;
	while ((opt = getopt(argc, argv, "s:")) != -1) {
		switch (opt) {
			case 's':
				size = atoi(optarg);
				break; 
			default:
				printf("invalid options, exiting"); 
				exit(1);
		}
	}

	
  	float a;// X[size], Y[size];
  	vector<float> res(size);
  	vector<float> X(size);
  	vector<float> Y(size); 
  	for (int i = 0; i < size; i++){
  		X[i] = i; Y[i] = size-i;
  	}
    a = 1.0; 
	printf("Computing SAXPY on with acc…:"); 
	double iStart = cpuSecond();
    acc_saxpy(res.data(), size, a, X.data(), Y.data()); 
	double accElaps = cpuSecond() - iStart;
	printf("Done! in:%fs \n", accElaps); 


  	vector<float> cX(size);
  	vector<float> cY(size); 
  	for (int i = 0; i < size; i++) {
  		cX[i] = i; cY[i] = size-i;
  	} 
   	printf("Computing SAXPY on the cpu…:"); 
	iStart = cpuSecond();
    cpu_saxpy(size, a, cX.data(), cY.data()); 
	double cElaps = cpuSecond() - iStart;
	printf("Done! in:%fs \n", cElaps); 

	a=0; 
    bool correct = true; 
    for(int i=0; i<size; ++i){
    	if(cY[i]!=res[i]){
    		printf("Error at:%d, %f!=%f\n", i, cY[i], res[i]); 
    		correct=false; 
    	}else{
    	//	printf("id: %d res:(%f==%f)\n", i, cY[i], res[i]); 
    	}
    }

    if(correct){
    	printf("\nverification passed!\n"); 
    }

    printf("%f\t%f\n", accElaps, cElaps); 


	return 0; 
}
