//
//  stacking.h
//  GMU-XVEC
//
//  Created by Jan Profant on 01/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#ifndef stacking_h
#define stacking_h

#include <iostream>
#include <string>
#include <vector>

#include "../nnet.hpp"


class StackingLayer : virtual public Layer {
public:
    StackingLayer(std::string name, std::vector<int> offsets) : offsets(offsets) {};
    float * forward(float *input, unsigned long num_samples, unsigned long num_dims);

private:
    std::vector<int> offsets;
};


#endif /* stacking_h */


float * StackingLayer::forward(float *input, unsigned long num_samples, unsigned long num_dims) {
    int maxval = offsets[std::max_element(offsets.begin(), offsets.end()) - offsets.begin()];
    int minval = offsets[std::min_element(offsets.begin(), offsets.end()) - offsets.begin()];
    
    return NULL;
}
