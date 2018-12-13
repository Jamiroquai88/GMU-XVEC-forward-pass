//
//  statistics_pooling.hpp
//  GMU-XVEC
//
//  Created by Jan Profant on 01/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#ifndef statistics_pooling_h
#define statistics_pooling_h


#include <iostream>
#include <string>
#include <vector>

#include "../nnet.hpp"


class StatisticsPoolingLayer : virtual public Layer {
public:
    StatisticsPoolingLayer(std::string name, int input_dim, bool output_stddevs, int left_context, int right_context, float variance_floor) : m_input_dim(input_dim), m_output_stddevs(output_stddevs), m_left_context(left_context), m_right_context(right_context), m_variance_floor(variance_floor) {};
    cl_mem forward(cl_mem input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context, cl_command_queue queue) ;
    
private:
    int m_input_dim;
    bool m_output_stddevs;
    int m_left_context;
    int m_right_context;
    float m_variance_floor;
    cl_mem m_output;
};


#endif /* statistics_pooling_h */
