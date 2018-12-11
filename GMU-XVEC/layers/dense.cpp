//
//  dense.cpp
//  GMU-XVEC
//
//  Created by Jan Profant on 08/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#include <stdio.h>

#include "dense.hpp"
#include "../utils.hpp"
#include "../opencl_utils.hpp"


DenseLayer::DenseLayer(std::string name, cl_context context, std::vector<float> linear, std::vector<float> bias) {
    cl_int err;
    linear_size = linear.size();
    cl_linear = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          sizeof(float) * linear.size(), &linear[0], &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    cl_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        sizeof(float) * bias.size(), &bias[0], &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
}


cl_mem DenseLayer::forward(cl_mem input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context, cl_command_queue queue) {
    size_t max_local_size;
    cl_int err;
    cl_program program;
    cl_kernel dot_kernel = compile_kernel(device, context, "layers/dense.cl", "dot_product2", max_local_size, program);
    
    unsigned long linear_cols = linear_size / cols;
    
    unsigned long output_rows = rows;
    unsigned long output_cols = linear_cols;
    unsigned long output_size = output_rows * output_cols;
    
    cl_mem output = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * output_rows * output_cols, NULL, &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    /* Create arguments for multiplication kernel */
    err = clSetKernelArg(dot_kernel, 0, sizeof(unsigned long), &output_size);
    err |= clSetKernelArg(dot_kernel, 1, sizeof(unsigned long), &rows);
    err |= clSetKernelArg(dot_kernel, 2, sizeof(unsigned long), &cols);
    err |= clSetKernelArg(dot_kernel, 3, sizeof(unsigned long), &linear_cols);
    err |= clSetKernelArg(dot_kernel, 4, sizeof(cl_mem), &input);
    err |= clSetKernelArg(dot_kernel, 5, sizeof(cl_mem), &cl_linear);
    err |= clSetKernelArg(dot_kernel, 6, sizeof(cl_mem), &cl_bias);
    err |= clSetKernelArg(dot_kernel, 7, sizeof(cl_mem), &output);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    /* Enqueue multiplication kernel */
    size_t global_size = get_global_group_size(output_size, max_local_size);
    err = clEnqueueNDRangeKernel(queue, dot_kernel, 1, NULL, &global_size,
                                 &max_local_size, 0, NULL, NULL);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    clReleaseMemObject(input);
    clReleaseKernel(dot_kernel);
    clReleaseProgram(program);
    
    rows = output_rows;
    cols = output_cols;
    return output;
}
