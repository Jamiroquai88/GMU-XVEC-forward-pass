//
//  stacking.cpp
//  GMU-XVEC
//
//  Created by Jan Profant on 08/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#include <stdio.h>

#include "stacking.hpp"
#include "../utils.hpp"
#include "../opencl_utils.hpp"


cl_mem StackingLayer::forward(cl_mem input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context, cl_command_queue queue) {
    int maxval = offsets[std::max_element(offsets.begin(), offsets.end()) - offsets.begin()];
    int minval = offsets[std::min_element(offsets.begin(), offsets.end()) - offsets.begin()];
    unsigned long output_rows = rows - (abs(minval) + maxval);
    unsigned long output_cols = offsets.size() * cols;
    unsigned long output_size = output_rows * output_cols;
    
    cl_program program;
    size_t max_local_size;
    cl_int err;
    cl_kernel stack_kernel = compile_kernel(device, context, "layers/stacking.cl", "stack", max_local_size, program);
    cl_mem output = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * output_size, NULL, &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    int pos0;
    unsigned long pos1;
    unsigned long start_row = maxval + abs(minval);
    for (unsigned long i = 0; i < offsets.size(); i++) {
        pos0 = abs(maxval) - offsets[i];
        pos1 = pos0 + rows;
        /* Create arguments for multiplication kernel */
        err = clSetKernelArg(stack_kernel, 0, sizeof(cl_mem), &input);
        err |= clSetKernelArg(stack_kernel, 1, sizeof(cl_mem), &output);
        err |= clSetKernelArg(stack_kernel, 2, sizeof(unsigned long), &cols);
        err |= clSetKernelArg(stack_kernel, 3, sizeof(unsigned long), &output_cols);
        err |= clSetKernelArg(stack_kernel, 4, sizeof(unsigned long), &output_rows);
        err |= clSetKernelArg(stack_kernel, 5, sizeof(unsigned long), &i);
        err |= clSetKernelArg(stack_kernel, 6, sizeof(int), &pos0);
        err |= clSetKernelArg(stack_kernel, 7, sizeof(unsigned long), &pos1);
        err |= clSetKernelArg(stack_kernel, 8, sizeof(unsigned long), &start_row);
        
        if (err < 0) {
            std::cerr << getCLError(err) << std::endl;
            exit(1);
        }
        
        /* Enqueue multiplication kernel */
        size_t global_size = get_global_group_size(output_size, max_local_size);
        err = clEnqueueNDRangeKernel(queue, stack_kernel, 1, NULL, &global_size,
                                     &max_local_size, 0, NULL, NULL);
        if (err < 0) {
            std::cerr << getCLError(err) << std::endl;
            exit(1);
        }
    }
    
    clReleaseMemObject(input);
    clReleaseKernel(stack_kernel);
    clReleaseProgram(program);
  
    rows = output_rows;
    cols = output_cols;
    return output;
}
