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
    cl_int err;
    m_kernel = compile_kernel(device, context, "layers/relu.cl", "activation", m_max_local_size, m_program);
    
    unsigned long output_size = rows * cols;
    
    /* Create arguments for activation kernel */
    err = clSetKernelArg(m_kernel, 0, sizeof(unsigned long), &output_size);
    err |= clSetKernelArg(m_kernel, 1, sizeof(cl_mem), &input);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    /* Enqueue multiplication kernel */
    m_global_size = get_global_group_size(output_size, m_max_local_size);
    err = clEnqueueNDRangeKernel(queue, m_kernel, 1, NULL, &m_global_size,
                                 &m_max_local_size, 0, NULL, &m_profiling_event);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    return input;
}
