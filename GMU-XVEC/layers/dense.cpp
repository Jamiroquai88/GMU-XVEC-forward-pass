//
//  dense.cpp
//  GMU-XVEC
//
//  Created by Jan Profant on 08/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#include <stdio.h>

#include "dense.hpp"
#include "../utils.hpp"
#include "../opencl_utils.hpp"


void DenseLayer::Free() {
    clReleaseMemObject(m_output);
}


DenseLayer::DenseLayer(std::string name, cl_context context, std::vector<float> linear, std::vector<float> bias) {
    cl_int err;
    m_linear_size = linear.size();
    m_linear = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          sizeof(float) * linear.size(), &linear[0], &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    m_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        sizeof(float) * bias.size(), &bias[0], &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
}


cl_mem DenseLayer::forward(cl_mem input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context, cl_command_queue queue) {
    cl_int err;
    m_kernel = compile_kernel(device, context, "layers/dense.cl", "dot_product2", m_max_local_size, m_program);
    
    unsigned long linear_cols = m_linear_size / cols;
    unsigned long output_rows = rows;
    unsigned long output_cols = linear_cols;
    unsigned long output_size = output_rows * output_cols;
    
    m_output = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * output_rows * output_cols, NULL, &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    /* Create arguments for multiplication kernel */
    err = clSetKernelArg(m_kernel, 0, sizeof(unsigned long), &output_size);
    err |= clSetKernelArg(m_kernel, 1, sizeof(unsigned long), &rows);
    err |= clSetKernelArg(m_kernel, 2, sizeof(unsigned long), &cols);
    err |= clSetKernelArg(m_kernel, 3, sizeof(unsigned long), &linear_cols);
    err |= clSetKernelArg(m_kernel, 4, sizeof(cl_mem), &input);
    err |= clSetKernelArg(m_kernel, 5, sizeof(cl_mem), &m_linear);
    err |= clSetKernelArg(m_kernel, 6, sizeof(cl_mem), &m_bias);
    err |= clSetKernelArg(m_kernel, 7, sizeof(cl_mem), &m_output);
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
    
    rows = output_rows;
    cols = output_cols;
    return m_output;
}
