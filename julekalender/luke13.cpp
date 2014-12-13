#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>
#include <time.h>

#define DATA_SIZE (1024)


const char *KernelSource =                                             "\n" \
"__kernel void mirp(                                                    \n" \
"   __global short* input,                                              \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i > 1000) {                                                      \n" \
"       output[i] = 0;                                                  \n" \
"       return;                                                         \n" \
"   }                                                                   \n" \
"   int j = get_global_id(0);                                           \n" \
"   int reverse = 0;                                                    \n" \
"   while (j != 0) {                                                    \n" \
"       reverse = reverse * 10;                                         \n" \
"       reverse = reverse + j%10;                                       \n" \
"       j = j/10;                                                       \n" \
"   }                                                                   \n" \
"   j = reverse;                                                        \n" \
"   output[i] = 1;                                                      \n" \
"   for (int k = 2; k < i; k++) {                                       \n" \
"       if (i % k == 0 && k != i) output[i] = 0;                        \n" \
"   }                                                                   \n" \
"   for (int k = 2; k < j; k++) {                                       \n" \
"       if (j % k == 0 && k != j) output[i] = 0;                        \n" \
"   }                                                                   \n" \
"   if(i-j == 0) output[i] = 0;                                         \n" \
"}                                                                      \n";\


int main(int argc, char** argv)
{
    int err;
    float results[DATA_SIZE];          
    unsigned int mirp;
    
    size_t global;                     
    size_t local;                      
    
    cl_device_id device_id;            
    cl_context context;                
    cl_command_queue commands;         
    cl_program program;                
    cl_kernel kernel;                  
    
    cl_mem input;                      
    cl_mem output;
    
    unsigned int count = DATA_SIZE;
    
    int gpu = 1;
    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);

    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "mirp", &err);
    input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(short) * count, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);

    clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    
    global = count;
    
    clock_t start;
    double duration;
    start = clock();

    
    clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    clFinish(commands);
    
    duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;
    printf("Tid: %f sekunder ", duration);

    clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * count, results, 0, NULL, NULL );

    
    mirp = 0;
    for(int i = 0; i < count; i++)
    {
        if(results[i] == 1) {
            printf("\n%d er mirptall", i);
            mirp++;
        }
    }
    

    printf("\n%d av %d tall er mirptall", mirp, count);

    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    
    return 0;
}