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


cl_mem ReLULayer::forward(cl_mem input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context, cl_command_queue queue) {
    size_t max_local_size;
    cl_int err;
    cl_program program;
    cl_kernel activation_kernel = compile_kernel(device, context, "layers/relu.cl", "activation", max_local_size, program);
    
    unsigned long output_size = rows * cols;
    
    /* Create arguments for activation kernel */
    err = clSetKernelArg(activation_kernel, 0, sizeof(unsigned long), &output_size);
    err |= clSetKernelArg(activation_kernel, 1, sizeof(cl_mem), &input);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    /* Enqueue multiplication kernel */
    size_t global_size = get_global_group_size(output_size, max_local_size);
    err = clEnqueueNDRangeKernel(queue, activation_kernel, 1, NULL, &global_size,
                                 &max_local_size, 0, NULL, NULL);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    clReleaseKernel(activation_kernel);
    clReleaseProgram(program);
    
    return input;
}
