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


void StatisticsExtractionLayer::Free() {
    clReleaseMemObject(m_output);
    clReleaseMemObject(m_input2);
}


cl_mem StatisticsExtractionLayer::forward(cl_mem input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context, cl_command_queue queue) {
    int variance_offset = m_include_variance ? 1 + (int)cols : -1;
    cols = m_include_variance ? 1 + 2 * cols : 1 + cols;
    unsigned long input_cols = variance_offset != -1 ? variance_offset - 1 : cols - 1;
    unsigned long output_size = rows * cols;
    std::vector<float> output(output_size);
    
    cl_int err;
    m_kernel = compile_kernel(device, context, "layers/statistics_extraction.cl", "extract_statistics", m_max_local_size, m_program);
    
    m_output = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * output_size, NULL, &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    m_input2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * output_size, NULL, &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    /* Create arguments for extraction kernel */
    err = clSetKernelArg(m_kernel, 0, sizeof(unsigned long), &cols);
    err |= clSetKernelArg(m_kernel, 1, sizeof(unsigned long), &rows);
    err |= clSetKernelArg(m_kernel, 2, sizeof(int), &variance_offset);
    err |= clSetKernelArg(m_kernel, 3, sizeof(unsigned long), &input_cols);
    err |= clSetKernelArg(m_kernel, 4, sizeof(cl_mem), &input);
    err |= clSetKernelArg(m_kernel, 5, sizeof(cl_mem), &m_input2);
    err |= clSetKernelArg(m_kernel, 6, sizeof(cl_mem), &m_output);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    /* Enqueue multiplication kernel */
    m_global_size = get_global_group_size(cols, m_max_local_size);
    err = clEnqueueNDRangeKernel(queue, m_kernel, 1, NULL, &m_global_size,
                                 &m_max_local_size, 0, NULL, &m_profiling_event);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    return m_output;
}
