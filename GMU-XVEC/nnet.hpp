//
//  nnet.hpp
//  GMU-XVEC
//
//  Created by Jan Profant on 28/11/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#ifndef nnet_hpp
#define nnet_hpp

#include <stdio.h>
#include <iostream>
#include <vector>
#include <unordered_map>


class Layer {
public:
    Layer();
};


class NNet {
public:
    NNet(std::string nnet_path);
private:
    std::vector<Layer> layers;
    std::unordered_map<std::string, std::string> ParseNodeAttributes(std::vector<std::string> attributes, std::string type);
    std::string ParseNodeAttributeValue(std::string value, std::vector<std::string> offsets);
};

#endif /* nnet_hpp */
