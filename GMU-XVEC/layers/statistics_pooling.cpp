//
//  statistics_pooling.cpp
//  GMU-XVEC
//
//  Created by Jan Profant on 09/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#include <stdio.h>
#include <math.h>

#include "../opencl_utils.hpp"
#include "statistics_pooling.hpp"


std::vector<float> StatisticsPoolingLayer::forward(std::vector<float> input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context) {
    assert(left_context == 0);
    assert(right_context > rows);
    std::vector<float> val(cols);
    std::copy(input.end() - cols, input.end(), val.begin());
    unsigned long dim = (cols - 1) / 2;
    std::vector<float> output(dim * 2);
    
    size_t max_local_size;
    cl_int err;
    cl_program program;
    cl_kernel extraction_kernel = compile_kernel(device, context, "layers/statistics_pooling.cl", "pool_statistics", max_local_size, program);
    
    // initialize input and output buffers
    cl_mem val_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * val.size(), &val[0], &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * output.size(), &output[0], &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    // initialize queue
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    /* Create arguments for extraction kernel */
    err |= clSetKernelArg(extraction_kernel, 0, sizeof(float), &variance_floor);
    err |= clSetKernelArg(extraction_kernel, 1, sizeof(unsigned long), &dim);
    err |= clSetKernelArg(extraction_kernel, 2, sizeof(unsigned long), &rows);
    err |= clSetKernelArg(extraction_kernel, 3, sizeof(cl_mem), &val_buffer);
    err |= clSetKernelArg(extraction_kernel, 4, sizeof(cl_mem), &output_buffer);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    /* Enqueue multiplication kernel */
    cl_event prof_event;
    size_t global_size = get_global_group_size(cols, max_local_size);
    err = clEnqueueNDRangeKernel(queue, extraction_kernel, 1, NULL, &global_size,
                                 &max_local_size, 0, NULL, &prof_event);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    /* Read output buffer */
    err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, sizeof(float) * output.size(), &output[0], 0, NULL, NULL);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    clReleaseMemObject(val_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseCommandQueue(queue);
    clReleaseKernel(extraction_kernel);
    clReleaseProgram(program);
    
    rows = 1;
    cols = dim * 2;
    return output;
}
