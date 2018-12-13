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

#include "../nnet.hpp"


class StatisticsExtractionLayer : virtual public Layer {
public:
    StatisticsExtractionLayer(std::string name, bool include_variance) : m_include_variance(include_variance) {};
    cl_mem forward(cl_mem input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context, cl_command_queue queue);
    void Free();
    
private:
    cl_mem m_output;
    cl_mem m_input;
    cl_mem m_input2;
    bool m_include_variance;
};


#endif /* statistics_extraction_h */
