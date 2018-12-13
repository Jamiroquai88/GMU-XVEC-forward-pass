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


cl_mem StatisticsPoolingLayer::forward(cl_mem input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context, cl_command_queue queue) {
    m_input = input;
    assert(m_left_context == 0);
    assert(m_right_context > rows);
    unsigned long offset = (rows - 1) * cols;
    unsigned long dim = (cols - 1) / 2;
    
    cl_int err;
    m_kernel = compile_kernel(device, context, "layers/statistics_pooling.cl", "pool_statistics", m_max_local_size, m_program);
    
    m_output = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * dim * 2, NULL, &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    /* Create arguments for extraction kernel */
    err |= clSetKernelArg(m_kernel, 0, sizeof(float), &m_variance_floor);
    err |= clSetKernelArg(m_kernel, 1, sizeof(unsigned long), &dim);
    err |= clSetKernelArg(m_kernel, 2, sizeof(unsigned long), &rows);
    err |= clSetKernelArg(m_kernel, 3, sizeof(cl_mem), &input);
    err |= clSetKernelArg(m_kernel, 4, sizeof(cl_mem), &m_output);
    err |= clSetKernelArg(m_kernel, 5, sizeof(unsigned long), &offset);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    /* Enqueue multiplication kernel */
    m_global_size = get_global_group_size(dim * 2, m_max_local_size);
    err = clEnqueueNDRangeKernel(queue, m_kernel, 1, NULL, &m_global_size,
                                 &m_max_local_size, 0, NULL, &m_profiling_event);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    rows = 1;
    cols = dim * 2;
    return m_output;
}
