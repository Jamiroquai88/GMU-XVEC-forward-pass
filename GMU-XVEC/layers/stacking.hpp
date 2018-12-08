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
#include <cmath>

#include "../nnet.hpp"


class StackingLayer : virtual public Layer {
public:
    StackingLayer(std::string name, std::vector<int> offsets) : offsets(offsets) {};
    std::vector<float> forward(std::vector<float> input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context);

private:
    std::vector<int> offsets;
};


#endif /* stacking_h */
