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


class StatisticsPoolingLayer : public Layer {
public:
    StatisticsPoolingLayer(std::string name, int input_dim, bool output_stddevs, int left_context, int right_context, float variance_floor) : Layer(name), input_dim(input_dim), output_stddevs(output_stddevs), left_context(left_context), right_context(right_context), variance_floor(variance_floor) {};
    
private:
    int input_dim;
    bool output_stddevs;
    int left_context;
    int right_context;
    float variance_floor;
};

#endif /* statistics_pooling_h */
