//
//  convolution.hpp
//  GMU-XVEC
//
//  Created by Jan Profant on 16/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#ifndef convolution_hpp
#define convolution_hpp

#include <stdio.h>
#include <vector>

#include "../opencl_utils.hpp"
#include "../nnet.hpp"


class ConvolutionalLayer : virtual public Layer {
public:
    ConvolutionalLayer(std::string name, cl_context context, unsigned long input_width, unsigned long input_height, unsigned long input_depth, unsigned long num_filters, unsigned long kernel_width, unsigned long kernel_height, unsigned long stride, std::vector<float> weights);
    cl_mem forward(cl_mem input, cl_device_id device, cl_context context, cl_command_queue queue);
    void Free();
    
private:
    unsigned long m_input_width;
    unsigned long m_input_height;
    unsigned long m_input_depth;
    unsigned long m_num_filters;
    unsigned long m_kernel_width;
    unsigned long m_kernel_height;
    unsigned long m_stride;
    cl_mem m_weights;
    unsigned long m_output_dim;
    cl_mem m_input;
    cl_mem m_output;
};

#endif /* convolution_hpp */
