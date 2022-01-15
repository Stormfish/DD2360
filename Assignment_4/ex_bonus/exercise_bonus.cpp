#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <iostream>
#include <getopt.h>

#include "utils.h"

#define CHK_ERROR(err) if (err != CL_SUCCESS) fprintf(stderr,"Error: %s\n",clGetErrorString(err));

constexpr int num_iterations = 10; 
int num_particles = 10000000; 

struct __attribute__ ((packed)) Particle
{
  cl_float x,y,z, vx,vy,vz; 
};



const char *particle_program =
"typedef struct __attribute__ ((packed)) Particle          \n"
"{                                                        \n"
"  float x,y,z, vx,vy,vz;                                 \n"
"}Particle;                                                       \n"
"                                                         \n"
"__kernel                                                 \n"
"void update(__global Particle* p, float dt, int num_particles){   \n"
"  const unsigned int tid = get_global_id(0);             \n"
"    if(tid >= num_particles){                            \n"
"      return;                                            \n"
"    }                                                    \n"
"                                                         \n"
"    //p-update                                           \n"
"  p[tid].x+=p[tid].x*dt;                             \n"
"  p[tid].y+=p[tid].y*dt;                             \n"
"  p[tid].z+=p[tid].z*dt;                             \n"
"  p[tid].vx=p[tid].vx*dt*0.50;                         \n"
"  p[tid].vy=p[tid].vy*dt*0.50;                         \n"
"  p[tid].vx=p[tid].vz*dt*0.50;                         \n"
"}                                                        \n";





void cpu_update(Particle* p, float dt){
  for(int i=0; i<num_particles; ++i){
    p[i].x+=p[i].x*dt;       
    p[i].y+=p[i].y*dt;       
    p[i].z+=p[i].z*dt;       
    p[i].vx=p[i].vx*dt*0.50; 
    p[i].vy=p[i].vy*dt*0.50; 
    p[i].vx=p[i].vz*dt*0.50; 
  } 
}

float r_bound(float max){
  float r = rand() / (static_cast<float>(RAND_MAX/max));
  return r; 
}


int main(int argc, char *argv[]) {
  uint block_size = 64; 

  int opt;
  while ((opt = getopt(argc, argv, "s:b:")) != -1) {
    switch (opt) {
      case 's':
        num_particles = atoi(optarg);
        break; 
      case 'b':
        block_size = atoi(optarg);
        break;
      default:
        printf("invalid options, exiting"); 
        exit(1);
    }
  }



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
  std::vector<Particle> pv(num_particles);
  srand(0); 
  for(auto& p:pv){
    p.x = {r_bound(1000)}; 
    p.y = {r_bound(1000)}; 
    p.z = {r_bound(1000)}; 
   
    p.vx = {r_bound(1000)}; 
    p.vy = {r_bound(1000)}; 
    p.vz = {r_bound(1000)};
  }

  /* Allocate device data */
  //cl_mem a_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, 1, NULL, &err);CHK_ERROR(err);
  cl_mem d_particles = clCreateBuffer(context, CL_MEM_READ_WRITE, num_particles*sizeof(Particle), NULL, &err);CHK_ERROR(err);

  /* Create the OpenCL program */
  cl_program program = clCreateProgramWithSource(context, 1,(const char **)&particle_program, NULL, &err);CHK_ERROR(err);

  /* Build code within and report any errors */
  err = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t len;
    char buffer[2048];
    clGetProgramBuildInfo(program, device_list[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len); 
    fprintf(stderr,"Build error: %s\n", buffer); exit(0);}

  /* Create a kernel object referncing our "saxpy" kernel */
  cl_kernel kernel = clCreateKernel(program, "update", &err);CHK_ERROR(err);

  /* Set the kernel arguments */
  cl_float dt = 1.0f; 
  cl_int cl_num = num_particles; 

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem),(void *)  &d_particles );CHK_ERROR(err);
  err = clSetKernelArg(kernel, 1, sizeof(cl_float),(void *)&dt );CHK_ERROR(err);
  err = clSetKernelArg(kernel, 2, sizeof(cl_int), (void *) &cl_num);CHK_ERROR(err);
  
  /* VSIZE work-items and one work-group */
  size_t bs_var = num_particles%block_size; 
  size_t n_workitem[1] = { num_particles+(block_size-bs_var) };
  size_t workgroup_size[1] = {block_size};

  //printf("bs, wi, rest = %u,%zu,%zu\n", block_size, *n_workitem, block_size-bs_var);
  //printf("Computing SAXPY on the GPU…:"); 
  /* Launch the kernel */
  double  iStart = cpuSecond();
  /* Send command to transfer host data to device */
  err = clEnqueueWriteBuffer(cmd_queue, d_particles, CL_TRUE, 0, num_particles*sizeof(Particle), pv.data(), 0, NULL, NULL);CHK_ERROR(err);

  for(int i=0; i<num_iterations; ++i){
    err = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, n_workitem, workgroup_size, 0, NULL, NULL);CHK_ERROR(err);
  }

  /* Transfer C vector back to host */
  err = clEnqueueReadBuffer(cmd_queue, d_particles, CL_TRUE, 0, num_particles*sizeof(Particle), pv.data(), 0, NULL, NULL);CHK_ERROR(err);
  /* Wait and make sure everything finishes */
  err = clFlush(cmd_queue);CHK_ERROR(err);
  err = clFinish(cmd_queue);CHK_ERROR(err);
  double gpuElaps = cpuSecond() - iStart;


  //printf("Done! in:%fs \n", gpuElaps);


  /* Check that result is correct */
  //CPU
  std::vector<Particle> pvc(num_particles);
  srand(0); 
  for(auto& p:pvc){
    p.x = {r_bound(1000)}; 
    p.y = {r_bound(1000)}; 
    p.z = {r_bound(1000)}; 
   
    p.vx = {r_bound(1000)}; 
    p.vy = {r_bound(1000)}; 
    p.vz = {r_bound(1000)};
  }
  //printf("Computing particles on the CPU…: ");
  iStart = cpuSecond();
  for(int i=0; i<num_iterations; ++i){
    cpu_update(pvc.data(), 1.0f); 
  }
  double cpuElaps = cpuSecond() - iStart;
  //printf("Done! in:%fs \n", cpuElaps); 

  //Compare, only pos x is fine
  float errsum = 0; 
  for(int i=0; i<num_particles; ++i){
    errsum+=pvc[i].x-pv[i].x; 
  }
  //printf("sum of errors: %f\n", errsum); 
  //printf("gpu(s)\t cpu(s)\n %f\t%f\n", gpuElaps, cpuElaps);
    printf("%f\t%f\n", gpuElaps, cpuElaps);

  //fprintf(stderr,"Program exiting...\n");

  /* Finally, release all that we have allocated. */
  err = clReleaseKernel(kernel);CHK_ERROR(err);
  err = clReleaseProgram(program);CHK_ERROR(err);
  err = clReleaseMemObject(d_particles);CHK_ERROR(err);
  err = clReleaseCommandQueue(cmd_queue);CHK_ERROR(err);
  err = clReleaseContext(context);CHK_ERROR(err);
  free(platforms);
  free(device_list);
  
  return 0;
}
