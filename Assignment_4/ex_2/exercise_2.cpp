#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "utils.h"

#define VSIZE 100000

#define CHK_ERROR(err) if (err != CL_SUCCESS) fprintf(stderr,"Error: %s\n",clGetErrorString(err));



const char *saxpy_program =
"__kernel                                       \n"
"void saxpy    (           int size,             \n"
"                         float a,              \n"
"                __global float *X,             \n"
"                __global float *Y)             \n"
"{                                              \n"
"    int index = get_global_id(0);              \n"
"    if(index<size)              \n"
"     Y[index] = a*X[index] + Y[index];          \n"
"  //  printf(\"Hello World! My threadId is:%d \\n \", index); \n"
"}                                              \n";


void cpu_saxpy(int size, float a, float* X, float* Y){
    for(int i=0; i<size; ++i){ 
        Y[i] = a*X[i] + Y[i]; 
    }
}



int main(int argc, char *argv[]) {
  cl_platform_id * platforms; cl_uint     n_platform;

  cl_int err = clGetPlatformIDs(0, NULL, &n_platform); CHK_ERROR(err);
  platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id)*n_platform);
  err = clGetPlatformIDs(n_platform, platforms, NULL); CHK_ERROR(err);

  cl_device_id *device_list; cl_uint n_devices;
  err = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, 0,NULL, &n_devices);CHK_ERROR(err);
  device_list = (cl_device_id *) malloc(sizeof(cl_device_id)*n_devices);
  err = clGetDeviceIDs( platforms[0],CL_DEVICE_TYPE_GPU, n_devices, device_list, NULL);CHK_ERROR(err);
  
  cl_context context = clCreateContext( NULL, n_devices, device_list, NULL, NULL, &err);CHK_ERROR(err);
  cl_command_queue cmd_queue = clCreateCommandQueue(context, device_list[0], 0, &err);CHK_ERROR(err);

  /* Initialize host memory/data */
  int array_size = VSIZE * sizeof(float);
  float X[VSIZE], Y[VSIZE];
  cl_float a=1; 
  int i;
  for ( i = 0; i < VSIZE; i++) 
    {X[i] = i; Y[i] = VSIZE-i;}
  
  
  /* Allocated device data */
  //cl_mem a_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, 1, NULL, &err);CHK_ERROR(err);
  cl_mem X_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, array_size, NULL, &err);CHK_ERROR(err);
  cl_mem Y_dev = clCreateBuffer(context, CL_MEM_WRITE_ONLY,array_size, NULL, &err);CHK_ERROR(err);
  
  /* Send command to transfer host data to device */
  err = clEnqueueWriteBuffer(cmd_queue, X_dev, CL_TRUE, 0, array_size, X, 0, NULL, NULL);CHK_ERROR(err);
  err = clEnqueueWriteBuffer(cmd_queue, Y_dev, CL_TRUE, 0, array_size, Y, 0, NULL, NULL);CHK_ERROR(err);

  /* Create the OpenCL program */
  cl_program program = clCreateProgramWithSource(context, 1,(const char **)&saxpy_program, NULL, &err);CHK_ERROR(err);
  
  /* Build code within and report any errors */
  err = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t len;
    char buffer[2048];
    clGetProgramBuildInfo(program, device_list[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len); 
    fprintf(stderr,"Build error: %s\n", buffer); exit(0);}
  
  /* Create a kernel object referncing our "saxpy" kernel */
  cl_kernel kernel = clCreateKernel(program, "saxpy", &err);CHK_ERROR(err);
  
  /* Set the four kernel arguments */
  cl_int size = VSIZE; 
  err = clSetKernelArg(kernel, 0, sizeof(cl_int),(void *)  &size );CHK_ERROR(err);
  err = clSetKernelArg(kernel, 1, sizeof(cl_float),(void *)&a );CHK_ERROR(err);
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &X_dev);CHK_ERROR(err);
  err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &Y_dev);CHK_ERROR(err);
  
  /* VSIZE work-items and one work-group */
  size_t bs_var = VSIZE%256; 
  size_t n_workitem[1] = { VSIZE+(256-bs_var) };
  size_t workgroup_size[1] = {256};

printf("bs, wi = %zu,%zu\n", bs_var, *n_workitem);
  printf("Computing SAXPY on the GPU…:"); 
  /* Launch the kernel */
  err = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, n_workitem, workgroup_size, 0, NULL, NULL);CHK_ERROR(err);
  
  /* Transfer C vector back to host */
  err = clEnqueueReadBuffer(cmd_queue, Y_dev, CL_TRUE, 0, array_size, Y, 0, NULL, NULL);CHK_ERROR(err);
  
  /* Wait and make sure everything finishes */
  err = clFlush(cmd_queue);CHK_ERROR(err);
  err = clFinish(cmd_queue);CHK_ERROR(err);
  printf("Done! \n"); 


  /* Check that result is correct */
  float ca, cX[VSIZE], cY[VSIZE];
  for ( int j = 0; j < VSIZE; j++) {
    cX[j] = j; 
    cY[j] = VSIZE-j;
  }
  ca=1.0f;
  printf("Computing SAXPY on the CPU…:"); 
  cpu_saxpy(VSIZE, ca, cX, cY);
  printf("Done! \n"); 
  printf("Comparing the output for each implementation… ");
  bool correct = true; 
  for ( i = 0; i < VSIZE; ++i){
    if (cY[i]!=Y[i]){
      fprintf(stderr,"Error at %d (%f /= %f)\n", i,cY[i],Y[i]);
      correct=false; 
    }else{}
  }
  if(correct)
      printf( " Correct!\n");

  fprintf(stderr,"Program exiting...\n");

  /* Finally, release all that we have allocated. */
  err = clReleaseKernel(kernel);CHK_ERROR(err);
  err = clReleaseProgram(program);CHK_ERROR(err);
  err = clReleaseMemObject(X_dev);CHK_ERROR(err);
  err = clReleaseMemObject(Y_dev);CHK_ERROR(err);
  err = clReleaseCommandQueue(cmd_queue);CHK_ERROR(err);
  err = clReleaseContext(context);CHK_ERROR(err);
  free(platforms);
  free(device_list);
  
  return 0;
}
