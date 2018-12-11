//
//  batchnorm.cpp
//  GMU-XVEC
//
//  Created by Jan Profant on 08/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#include <stdio.h>

#include "batchnorm.hpp"
#include "../opencl_utils.hpp"


BatchNormLayer::BatchNormLayer(std::string name, cl_context context, std::vector<float> mean, std::vector<float> variance, float epsilon) {
    cl_int err;
    // initialize mean, variance and epsilon buffers
    cl_mean = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * mean.size(), &mean[0], &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    cl_variance = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * variance.size(), &variance[0], &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    cl_epsilon = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float), &epsilon, &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
}


cl_mem BatchNormLayer::forward(cl_mem input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context, cl_command_queue queue) {
    size_t max_local_size;
    cl_int err;
    cl_program program;
    cl_kernel batchnorm_kernel = compile_kernel(device, context, "layers/batchnorm.cl", "batchnorm", max_local_size, program);
    unsigned long output_size = cols * rows;
    
    /* Create arguments for activation kernel */
    err = clSetKernelArg(batchnorm_kernel, 0, sizeof(unsigned long), &output_size);
    err = clSetKernelArg(batchnorm_kernel, 1, sizeof(unsigned long), &cols);
    err |= clSetKernelArg(batchnorm_kernel, 2, sizeof(cl_mem), &cl_mean);
    err |= clSetKernelArg(batchnorm_kernel, 3, sizeof(cl_mem), &cl_variance);
    err |= clSetKernelArg(batchnorm_kernel, 4, sizeof(cl_mem), &cl_epsilon);
    err |= clSetKernelArg(batchnorm_kernel, 5, sizeof(cl_mem), &input);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
        
    /* Enqueue multiplication kernel */
    size_t global_size = get_global_group_size(output_size, max_local_size);
    err = clEnqueueNDRangeKernel(queue, batchnorm_kernel, 1, NULL, &global_size,
                                 &max_local_size, 0, NULL, NULL);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    clReleaseKernel(batchnorm_kernel);
    clReleaseProgram(program);
    
    return input;
}
