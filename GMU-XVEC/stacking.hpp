//
//  stacking.h
//  GMU-XVEC
//
//  Created by Jan Profant on 01/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#ifndef stacking_h
#define stacking_h


class StackingLayer : public Layer {
public:
    StackingLayer(std::string name, std::vector<int> offsets) {this->offsets = offsets;}

private:
    std::vector<int> offsets;
};

//
//StackingLayer::StackingLayer(std::string name, std::vector<int> offsets) {
//    this->offsets = offsets;
//}

#endif /* stacking_h */
