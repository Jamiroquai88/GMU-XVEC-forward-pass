//
//  statistics_extraction.hpp
//  GMU-XVEC
//
//  Created by Jan Profant on 01/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#ifndef statistics_extraction_h
#define statistics_extraction_h


#include <iostream>
#include <string>
#include <vector>

#include "nnet.hpp"


class StatisticsExtractionLayer : public Layer {
public:
    StatisticsExtractionLayer(std::string name, bool include_variance) : Layer(name), include_variance(include_variance) {};
    
private:
    bool include_variance;
};

#endif /* statistics_extraction_h */
