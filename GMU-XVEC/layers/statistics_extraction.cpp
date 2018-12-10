//
//  statistics_extraction.cpp
//  GMU-XVEC
//
//  Created by Jan Profant on 09/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#include <stdio.h>
#include <numeric>
#include <vector>
#include <iostream>
#include <iterator>
#include <functional>

#include "../opencl_utils.hpp"
#include "statistics_extraction.hpp"


std::vector<float> StatisticsExtractionLayer::forward(std::vector<float> input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context) {
    unsigned long variance_offset = include_variance ? 1 + cols : 0;
    cols = include_variance ? 1 + 2 * cols : 1 + cols;
    unsigned long output_size = rows * cols;
    std::vector<float> input2(output_size, 1.0f);
    std::vector<float> output(output_size);
    for (unsigned int i = 0; i < rows; i++)
        std::copy(input.begin() + i * (cols - variance_offset), input.begin() + (i + 1) * (cols - variance_offset), input2.begin() + 1 + i * cols);
    
    size_t max_local_size;
    cl_int err;
    cl_program program;
    cl_kernel extraction_kernel = compile_kernel(device, context, "layers/statistics_extraction.cl", "extract_statistics", max_local_size, program);
    
    // initialize input an output buffers
    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * output_size, &input2[0], &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * output_size, &output[0], &err);
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
    err = clSetKernelArg(extraction_kernel, 0, sizeof(unsigned long), &cols);
    err |= clSetKernelArg(extraction_kernel, 1, sizeof(unsigned long), &rows);
    err |= clSetKernelArg(extraction_kernel, 2, sizeof(unsigned long), &variance_offset);
    err |= clSetKernelArg(extraction_kernel, 3, sizeof(cl_mem), &input_buffer);
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
    err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, sizeof(float) * output_size, &output[0], 0, NULL, NULL);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseCommandQueue(queue);
    clReleaseKernel(extraction_kernel);
    clReleaseProgram(program);
    return output;
}
