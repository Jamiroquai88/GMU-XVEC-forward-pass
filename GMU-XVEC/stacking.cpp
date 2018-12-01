//
//  stacking.cpp
//  GMU-XVEC
//
//  Created by Jan Profant on 01/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#include <stdio.h>

#include "stacking.hpp"


StackingLayer::StackingLayer(std::string name, std::vector<int> offsets) {
    this->offsets = offsets;
}
