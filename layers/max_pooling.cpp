//
//  max_pooling.cpp
//  GMU-XVEC
//
//  Created by Jan Profant on 27/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#include "max_pooling.hpp"
#include "../opencl_utils.hpp"


cl_mem MaxPoolingLayer::forward(cl_mem input, cl_device_id device, cl_context context, cl_command_queue queue) {
    m_kernel = compile_kernel(device, context, "layers/max_pooling.cl", "max_pool", m_max_local_size, m_program);
    
    cl_int err;
    unsigned long output_height = (m_height - m_filter_dim) / m_stride + 1;
    unsigned long output_width = (m_width - m_filter_dim) / m_stride + 1;
    unsigned long output_size = output_height * output_width * m_depth;
    m_output = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * output_size, NULL, &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    /* Create arguments for activation kernel */
    err = clSetKernelArg(m_kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(m_kernel, 1, sizeof(cl_mem), &m_output);
    err |= clSetKernelArg(m_kernel, 2, sizeof(unsigned long), &m_filter_dim);
    err |= clSetKernelArg(m_kernel, 3, sizeof(unsigned long), &m_stride);
    err |= clSetKernelArg(m_kernel, 4, sizeof(unsigned long), &m_height);
    err |= clSetKernelArg(m_kernel, 5, sizeof(unsigned long), &m_width);
    err |= clSetKernelArg(m_kernel, 6, sizeof(unsigned long), &m_depth);
    err |= clSetKernelArg(m_kernel, 7, sizeof(unsigned long), &output_size);
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
    m_input = input;
    return m_output;
}
