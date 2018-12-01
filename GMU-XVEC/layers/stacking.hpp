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


class StackingLayer : public Layer {
public:
    StackingLayer(std::string name, std::vector<int> offsets) : Layer(name), offsets(offsets) {};

private:
    std::vector<int> offsets;
};


#endif /* stacking_h */
