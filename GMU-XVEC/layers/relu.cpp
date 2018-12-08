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


std::vector<float> forward(std::vector<float> input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context) {
    size_t max_local_size;
    cl_int err;
    cl_kernel dot_kernel = compile_kernel(device, context, "layers/dense.cl", "dot_product", max_local_size);
    
    std::vector<float> output(rows * cols, 0.0f);
    unsigned long output_size = output.size();
    
    // initialize input buffer
    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(float) * output_size, &input[0], &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    // initialize output buffer
    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * output_size, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    /* Create arguments for activation kernel */
    err = clSetKernelArg(dot_kernel, 0, sizeof(cl_mem), &input_buffer);
    err |= clSetKernelArg(dot_kernel, 1, sizeof(cl_mem), &output_buffer);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    /* Enqueue multiplication kernel */
    cl_event prof_event;
    size_t global_size = output_size;
    err = clEnqueueNDRangeKernel(queue, dot_kernel, 1, NULL, &global_size,
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
    
    return output;
}
