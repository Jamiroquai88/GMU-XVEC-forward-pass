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

#define MAC
#ifdef MAC
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif


class Layer {
public:
    
    virtual void foo() {};
    void ProfileInfo(std::string type);
    void FreeBase();
    
    std::string m_name;
    cl_program m_program;
    cl_kernel m_kernel;
    size_t m_max_local_size;
    size_t m_global_size;
    cl_event m_profiling_event;
};


class NNet {
public:
    NNet(std::string nnet_path, cl_context context);
    std::vector<float> forward(std::string fea_path, cl_device_id device, cl_context context, cl_command_queue queue, unsigned long &rows, unsigned long &cols);
    
private:
    std::unordered_map<std::string, std::string> ParseNodeAttributes(std::vector<std::string> attributes, std::string type);
    std::string ParseNodeAttributeValue(std::string value, std::vector<std::string> &offsets);
    std::unordered_map<std::string, std::string> ParseComponentAttributes(                                                                          std::vector<std::string> line_split, std::string &matrix_key, bool &end, std::unordered_map<std::string, std::vector<float>> &matrix_attrs);
    void ParseFloatsLine(std::vector<float> &matrix, std::vector<std::string> line_split, bool &matrix_end);
    
    void InitLayersFromNode(std::unordered_map<std::string, std::string> &node_attrs, std::unordered_map<std::string, std::vector<float>> &matrix_attrs);
    
    void FreeOutputs();
    
    std::vector<Layer*> m_layers;
    std::vector<std::string> m_layers_types;
    cl_context m_context;
};

#endif /* nnet_hpp */
