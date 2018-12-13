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
    m_input = input;
    int maxval = m_offsets[std::max_element(m_offsets.begin(), m_offsets.end()) - m_offsets.begin()];
    int minval = m_offsets[std::min_element(m_offsets.begin(), m_offsets.end()) - m_offsets.begin()];
    unsigned long output_rows = rows - (abs(minval) + maxval);
    unsigned long output_cols = m_offsets.size() * cols;
    unsigned long output_size = output_rows * output_cols;
    
    cl_int err;
    m_kernel = compile_kernel(device, context, "layers/stacking.cl", "stack", m_max_local_size, m_program);
    m_output = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * output_size, NULL, &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    int pos0;
    unsigned long pos1;
    unsigned long start_row = maxval + abs(minval);
    for (unsigned long i = 0; i < m_offsets.size(); i++) {
        pos0 = abs(maxval) - m_offsets[i];
        pos1 = pos0 + rows;
        /* Create arguments for multiplication kernel */
        err = clSetKernelArg(m_kernel, 0, sizeof(cl_mem), &input);
        err |= clSetKernelArg(m_kernel, 1, sizeof(cl_mem), &m_output);
        err |= clSetKernelArg(m_kernel, 2, sizeof(unsigned long), &cols);
        err |= clSetKernelArg(m_kernel, 3, sizeof(unsigned long), &output_cols);
        err |= clSetKernelArg(m_kernel, 4, sizeof(unsigned long), &output_rows);
        err |= clSetKernelArg(m_kernel, 5, sizeof(unsigned long), &i);
        err |= clSetKernelArg(m_kernel, 6, sizeof(int), &pos0);
        err |= clSetKernelArg(m_kernel, 7, sizeof(unsigned long), &pos1);
        err |= clSetKernelArg(m_kernel, 8, sizeof(unsigned long), &start_row);
        
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
    }
  
    rows = output_rows;
    cols = output_cols;
    return m_output;
}
