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


class DenseLayer : virtual public Layer {
public:
    DenseLayer(std::string name, float *linear, float *bias) : linear(linear), bias(bias) {};
    float * forward(float *input, unsigned long num_samples, unsigned long num_dims);
    
private:
    float *linear;
    float *bias;
};


float * DenseLayer::forward(float *input, unsigned long num_samples, unsigned long num_dims) {
    return NULL;
}

#endif /* dense_h */
