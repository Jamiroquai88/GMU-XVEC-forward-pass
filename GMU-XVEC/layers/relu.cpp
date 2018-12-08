//
//  relu.cpp
//  GMU-XVEC
//
//  Created by Jan Profant on 08/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#include <stdio.h>

#include "relu.hpp"
#include "../opencl_utils.hpp"


std::vector<float> ReLULayer::forward(std::vector<float> input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context) {
    size_t max_local_size;
    cl_int err;
    cl_kernel activation_kernel = compile_kernel(device, context, "layers/relu.cl", "activation", max_local_size);
    
    std::vector<float> output(rows * cols, 0.0f);
    unsigned long output_size = output.size();
    
    // initialize input buffer
    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         sizeof(float) * output_size, &input[0], &err);
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
    
    /* Create arguments for activation kernel */
    err = clSetKernelArg(activation_kernel, 0, sizeof(unsigned long), &output_size);
    err |= clSetKernelArg(activation_kernel, 1, sizeof(cl_mem), &input_buffer);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    /* Enqueue multiplication kernel */
    cl_event prof_event;
    size_t global_size = max_local_size * 4;
    err = clEnqueueNDRangeKernel(queue, activation_kernel, 1, NULL, &global_size,
                                 &max_local_size, 0, NULL, &prof_event);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    /* Read output buffer */
    err = clEnqueueReadBuffer(queue, input_buffer, CL_TRUE, 0, sizeof(float) * output_size, &output[0], 0, NULL, NULL);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    return output;
}
