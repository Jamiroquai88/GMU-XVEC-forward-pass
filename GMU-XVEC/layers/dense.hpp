//
//  dense.hpp
//  GMU-XVEC
//
//  Created by Jan Profant on 01/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#ifndef dense_h
#define dense_h

#include <iostream>
#include <string>
#include <vector>

#include "../nnet.hpp"


class DenseLayer : public Layer {
public:
    DenseLayer(std::string name, float *linear, float *bias) : Layer(name), linear(linear), bias(bias) {};
    float * forward(float *input, int num_samples, int num_dims);
    
private:
    float *linear;
    float *bias;
};


float * DenseLayer::forward(float *input, int num_samples, int num_dims) {
    return NULL;
}

#endif /* dense_h */
