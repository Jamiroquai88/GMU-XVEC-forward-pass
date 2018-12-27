//
//  max_pooling.hpp
//  GMU-XVEC
//
//  Created by Jan Profant on 27/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#ifndef max_pooling_hpp
#define max_pooling_hpp

#include <stdio.h>
#include <string>

#include "../nnet.hpp"


class MaxPoolingLayer : virtual public Layer {
public:
    MaxPoolingLayer(std::string name, unsigned long height, unsigned long width, unsigned long depth, unsigned long filter_dim=2, unsigned long stride=2) : m_height(height), m_width(width), m_depth(depth), m_filter_dim(filter_dim), m_stride(stride) {};
    cl_mem forward(cl_mem input, cl_device_id device, cl_context context, cl_command_queue queue);
    void Free() {clReleaseMemObject(m_input);};
    
private:
    unsigned long m_filter_dim;
    unsigned long m_stride;
    unsigned long m_height;
    unsigned long m_width;
    unsigned long m_depth;
    cl_mem m_input;
    cl_mem m_output;
};

#endif /* max_pooling_hpp */
