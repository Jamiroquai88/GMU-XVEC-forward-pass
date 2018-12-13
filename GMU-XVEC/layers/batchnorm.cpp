//
//  batchnorm.cpp
//  GMU-XVEC
//
//  Created by Jan Profant on 08/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#include <stdio.h>

#include "batchnorm.hpp"
#include "../opencl_utils.hpp"


BatchNormLayer::BatchNormLayer(std::string name, cl_context context, std::vector<float> mean, std::vector<float> variance, float epsilon) {
    cl_int err;
    // initialize mean, variance and epsilon buffers
    m_mean = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * mean.size(), &mean[0], &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    m_variance = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * variance.size(), &variance[0], &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    m_epsilon = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float), &epsilon, &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
}


cl_mem BatchNormLayer::forward(cl_mem input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context, cl_command_queue queue) {
    cl_int err;
    m_kernel = compile_kernel(device, context, "layers/batchnorm.cl", "batchnorm", m_max_local_size, m_program);
    unsigned long output_size = cols * rows;
    
    /* Create arguments for activation kernel */
    err = clSetKernelArg(m_kernel, 0, sizeof(unsigned long), &output_size);
    err = clSetKernelArg(m_kernel, 1, sizeof(unsigned long), &cols);
    err |= clSetKernelArg(m_kernel, 2, sizeof(cl_mem), &m_mean);
    err |= clSetKernelArg(m_kernel, 3, sizeof(cl_mem), &m_variance);
    err |= clSetKernelArg(m_kernel, 4, sizeof(cl_mem), &m_epsilon);
    err |= clSetKernelArg(m_kernel, 5, sizeof(cl_mem), &input);
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
