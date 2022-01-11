#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "utils.h"

#define VSIZE 1024

#define CHK_ERROR(err) if (err != CL_SUCCESS) fprintf(stderr,"Error: %s\n",clGetErrorString(err));



const char *vadd_program =
"__kernel                                       \n"
"void vadd    (  __global float *A,             \n"
"                __global float *B,             \n"
"                __global float *C)             \n"
"{                                              \n"
"    int index = get_global_id(0);              \n"
"    C[index] = A[index] + B[index];            \n"
"    printf(\"Hello World! My threadId is:%d \\n \", index); \n"
"}                                              \n";



int main(int argc, char *argv) {
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
  float A[VSIZE], B[VSIZE], C[VSIZE];
  int i;
  for ( i = 0; i < VSIZE; i++) 
    {A[i] = i; B[i] = VSIZE-i; C[i] = 0; }
  
  /* Allocated device data */
  cl_mem A_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, array_size, NULL, &err);CHK_ERROR(err);
  cl_mem B_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, array_size, NULL, &err);CHK_ERROR(err);
  cl_mem C_dev = clCreateBuffer(context, CL_MEM_WRITE_ONLY,array_size, NULL, &err);CHK_ERROR(err);
  
  /* Send command to transfer host data to device */
  err = clEnqueueWriteBuffer(cmd_queue, A_dev, CL_TRUE, 0, array_size, A, 0, NULL, NULL);CHK_ERROR(err);
  err = clEnqueueWriteBuffer(cmd_queue, B_dev, CL_TRUE, 0, array_size, B, 0, NULL, NULL);CHK_ERROR(err);

  /* Create the OpenCL program */
  cl_program program = clCreateProgramWithSource(context, 1,(const char **)&vadd_program, NULL, &err);CHK_ERROR(err);
  
  /* Build code within and report any errors */
  err = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t len;
    char buffer[2048];
    clGetProgramBuildInfo(program, device_list[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len); 
    fprintf(stderr,"Build error: %s\n", buffer); exit(0);}
  
  /* Create a kernel object referncing our "vadd" kernel */
  cl_kernel kernel = clCreateKernel(program, "vadd", &err);CHK_ERROR(err);
  
  /* Set the three kernel arguments */
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &A_dev);CHK_ERROR(err);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &B_dev);CHK_ERROR(err);
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &C_dev);CHK_ERROR(err);
  
  /* VSIZE work-items and one work-group */
  size_t n_workitem[1] = {VSIZE};
  size_t workgroup_size[1] = {1};

  /* Launch the kernel */
  cl_event event;
  err = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, n_workitem, workgroup_size, 0, NULL, NULL);CHK_ERROR(err);
  
  /* Transfer C vector back to host */
  err = clEnqueueReadBuffer(cmd_queue, C_dev, CL_TRUE, 0, array_size, C, 0, NULL, NULL);CHK_ERROR(err);
  
  /* Wait and make sure everything finishes */
  err = clFlush(cmd_queue);CHK_ERROR(err);
  err = clFinish(cmd_queue);CHK_ERROR(err);



  /* Check that result is correct */
    for ( i = 0; i < VSIZE; i++){ 
      /* Initialize host memory/data */
      int array_size = VSIZE * sizeof(float);
      float A[VSIZE], B[VSIZE], C[VSIZE];
      int i;
      for ( i = 0; i < VSIZE; i++) 
        {A[i] = i; B[i] = VSIZE-i; C[i] = 0; }

        if (C[i] != A[i]+B[i]){
            fprintf(stderr,"Error at %d (%f /= %f)\n", i,C[i],A[i]+B[i]);
        }
    }
  fprintf(stderr,"Program executed correctly...\n");

  /* Finally, release all that we have allocated. */
  err = clReleaseKernel(kernel);CHK_ERROR(err);
  err = clReleaseProgram(program);CHK_ERROR(err);
  err = clReleaseMemObject(A_dev);CHK_ERROR(err);
  err = clReleaseMemObject(B_dev);CHK_ERROR(err);
  err = clReleaseMemObject(C_dev);CHK_ERROR(err);
  err = clReleaseCommandQueue(cmd_queue);CHK_ERROR(err);
  err = clReleaseContext(context);CHK_ERROR(err);
  free(platforms);
  free(device_list);
  
  return 0;
}