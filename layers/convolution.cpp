//
//  convolution.cpp
//  GMU-XVEC
//
//  Created by Jan Profant on 16/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#include "convolution.hpp"


ConvolutionalLayer::ConvolutionalLayer(std::string name, cl_context context, unsigned long input_width, unsigned long input_height, unsigned long input_depth, unsigned long filters, unsigned long kernel_width, unsigned long kernel_height, unsigned long stride, std::vector<float> weights) : m_input_width(input_width), m_input_height(input_height), m_input_depth(input_depth), m_num_filters(filters), m_kernel_width(kernel_width), m_kernel_height(kernel_height), m_stride(stride)
{
    if (input_width != input_height) {
        std::cerr << "Invalid width and height of input images." << std::endl;
        exit(1);
    }
    if (kernel_width != kernel_height) {
        std::cerr << "Invalid width and height of kernel." << std::endl;
        exit(1);
    }
    unsigned long expected_weights_size = m_num_filters * input_depth * kernel_width * kernel_height;
    if (weights.size() != expected_weights_size) {
        std::cerr << "Invalid size of weights, got " << weights.size() << " instead of " << expected_weights_size << "." << std::endl;
        exit(1);
    }
    else {
        cl_int err;
        m_weights = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * weights.size(), &weights[0], &err);
        if (err < 0) {
            std::cerr << getCLError(err) << std::endl;
            exit(1);
        }
    }
    m_output_dim = (input_width - kernel_width) / (float)stride + 1;
}


void ConvolutionalLayer::Free() {
    clReleaseMemObject(m_weights);
    clReleaseMemObject(m_input);
}


cl_mem ConvolutionalLayer::forward(cl_mem input, cl_device_id device, cl_context context, cl_command_queue queue) {
    m_kernel = compile_kernel(device, context, "layers/convolution.cl", "convolve", m_max_local_size, m_program);
    
    cl_int err;
    unsigned long output_size = m_output_dim * m_output_dim * m_num_filters;
    m_output = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * output_size, NULL, &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    /* Create arguments for activation kernel */
    err = clSetKernelArg(m_kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(m_kernel, 1, sizeof(cl_mem), &m_output);
    err |= clSetKernelArg(m_kernel, 2, sizeof(cl_mem), &m_weights);
    err |= clSetKernelArg(m_kernel, 3, sizeof(unsigned long), &m_input_width);
    err |= clSetKernelArg(m_kernel, 4, sizeof(unsigned long), &m_output_dim);
    err |= clSetKernelArg(m_kernel, 5, sizeof(unsigned long), &m_input_depth);
    err |= clSetKernelArg(m_kernel, 6, sizeof(unsigned long), &m_num_filters);
    err |= clSetKernelArg(m_kernel, 7, sizeof(unsigned long), &m_kernel_width);
    err |= clSetKernelArg(m_kernel, 8, sizeof(unsigned long), &m_stride);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    /* Enqueue multiplication kernel */
    m_global_size = get_global_group_size(output_size / m_num_filters, m_max_local_size);
    err = clEnqueueNDRangeKernel(queue, m_kernel, 1, NULL, &m_global_size,
                                 &m_max_local_size, 0, NULL, &m_profiling_event);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    m_input = input;
    return m_output;
}

