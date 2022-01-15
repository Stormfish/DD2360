#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "utils.h"

#define VSIZE 1024

#define CHK_ERROR(err) if (err != CL_SUCCESS) fprintf(stderr,"Error: %s\n",clGetErrorString(err));



const char *v_program =
"__kernel                                       \n"
"void v    (                   )             \n"
"{                                              \n"
"    int index = get_global_id(0);              \n"
"    printf(\"Hello World! My threadId is:%d \\n \", index); \n"
"}                                              \n";




int main(int argc, char *argv) {
  cl_platform_id * platforms; cl_uint     n_platform;

  // Find OpenCL Platforms
  cl_int err = clGetPlatformIDs(0, NULL, &n_platform); CHK_ERROR(err);
  platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id)*n_platform);
  err = clGetPlatformIDs(n_platform, platforms, NULL); CHK_ERROR(err);

  // Find and sort devices
  cl_device_id *device_list; cl_uint n_devices;
  err = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, 0,NULL, &n_devices);CHK_ERROR(err);
  device_list = (cl_device_id *) malloc(sizeof(cl_device_id)*n_devices);
  err = clGetDeviceIDs( platforms[0],CL_DEVICE_TYPE_GPU, n_devices, device_list, NULL);CHK_ERROR(err);
  
  // Create and initialize an OpenCL context
  cl_context context = clCreateContext( NULL, n_devices, device_list, NULL, NULL, &err);CHK_ERROR(err);

  // Create a command queue
  cl_command_queue cmd_queue = clCreateCommandQueue(context, device_list[0], 0, &err);CHK_ERROR(err); 

  /* Insert your own code here */
    /* Create the OpenCL program */
  cl_program program = clCreateProgramWithSource(context, 1,(const char **)&v_program, NULL, &err);CHK_ERROR(err);
  
    /* Build code within and report any errors */
  err = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t len;
    char buffer[2048];
    clGetProgramBuildInfo(program, device_list[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len); 
    fprintf(stderr,"Build error: %s\n", buffer); exit(0);
  }
  
    /* Create a kernel object referncing our "vadd" kernel */
  cl_kernel kernel = clCreateKernel(program, "v", &err);CHK_ERROR(err);
  
  /* VSIZE work-items and one work-group */
  size_t n_workitem[1] = {VSIZE};
  size_t workgroup_size[1] = {1};
  err = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, n_workitem, workgroup_size, 0, NULL, NULL);CHK_ERROR(err);
  
  /* Wait and make sure everything finishes */
  err = clFlush(cmd_queue);CHK_ERROR(err);
  err = clFinish(cmd_queue);CHK_ERROR(err);

  
  // Finally, release all that we have allocated.
  err = clReleaseKernel(kernel);CHK_ERROR(err);
  err = clReleaseProgram(program);CHK_ERROR(err);
  err = clReleaseCommandQueue(cmd_queue);CHK_ERROR(err);
  err = clReleaseContext(context);CHK_ERROR(err);
  free(platforms);
  free(device_list);
  
  return 0;
}
