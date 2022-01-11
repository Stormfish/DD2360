#include <iostream>
#include <vector>
#include "utils.h"
constexpr int size = 400000000; 

using std::vector;
/*void matMulAcc(float *P, const float* M, const float* N, int Mh, int Mw, int Nw){
	

	#pragma acc parallel loop copyin(M[0:Mh*Mw]) copyin(N[0:Nw*Mw]) copyout(P[0:Mh*Nw])
	for(int i=0; i<Mh; ++i){
		#pragma acc loop
		for(int j=0; j<Nw; ++j){
			float sum=0.0; 
			for(int k=0; k<Mw; ++k){
				float a = M[i*Mw+k]; 
				float b = N[k*Nw+j]; 
				sum += a*b; 
			}
			P[i*Nw+j] = sum; 
		}
	}
}*/

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




int main(){

	
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
	double iElaps = cpuSecond() - iStart;
	printf("Done! in:%fs \n", iElaps); 


  	vector<float> cX(size);
  	vector<float> cY(size); 
  	for (int i = 0; i < size; i++) {
  		cX[i] = i; cY[i] = size-i;
  	} 
   	printf("Computing SAXPY on the cpu…:"); 
	iStart = cpuSecond();
    cpu_saxpy(size, a, cX.data(), cY.data()); 
	iElaps = cpuSecond() - iStart;
	printf("Done! in:%fs \n", iElaps); 

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


	return 0; 
}
