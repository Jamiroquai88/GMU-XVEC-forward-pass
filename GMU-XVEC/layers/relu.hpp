//
//  relu.hpp
//  GMU-XVEC
//
//  Created by Jan Profant on 01/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#ifndef relu_h
#define relu_h

#include <iostream>
#include <string>
#include <vector>

#include "../nnet.hpp"


class ReLULayer : virtual public Layer {
public:
    ReLULayer(std::string name) {};
    float * forward(float *input, unsigned long num_samples, unsigned long num_dims);
    
};


float * ReLULayer::forward(float *input, unsigned long num_samples, unsigned long num_dims) {
    return NULL;
}


#endif /* relu_h */
