//
//  statistics_pooling.cpp
//  GMU-XVEC
//
//  Created by Jan Profant on 09/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#include <stdio.h>

#include "statistics_pooling.hpp"


std::vector<float> StatisticsPoolingLayer::forward(std::vector<float> input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context) {
    assert(rows == 1);
    std::vector<float> output(cols * 2, 0.0f);
    std::copy(input.begin(), input.end(), output.begin() + cols);
    
    
    return output;
}
