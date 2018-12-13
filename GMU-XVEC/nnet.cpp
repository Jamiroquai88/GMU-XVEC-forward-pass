//
//  nnet.cpp
//  GMU-XVEC
//
//  Created by Jan Profant on 28/11/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//


#include <fstream>
#include <vector>
#include <unordered_map>
#include <regex>
#include <set>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>

#include "opencl_utils.hpp"
#include "nnet.hpp"
#include "utils.hpp"
#include "layers/stacking.hpp"
#include "layers/dense.hpp"
#include "layers/relu.hpp"
#include "layers/batchnorm.hpp"
#include "layers/statistics_extraction.hpp"
#include "layers/statistics_pooling.hpp"


NNet::NNet(std::string nnet_path, cl_context context) {
    m_context = context;
    std::cout << "Parsing neural net config from file " << nnet_path << "." << std::endl;
    std::string line;
    std::vector<std::string> lines;
    std::ifstream infile(nnet_path);
    if (!infile.good()) {
        std::cerr << "Path to nnet file does not exist: " << nnet_path << std::endl;
        exit(1);
    }
    
    while (std::getline(infile, line)) {
        lines.push_back(line);
    }
    
    unsigned long lines_num = lines.size();
    
    if (lines[0] != "<Nnet3> " || lines[lines_num - 1] != "</Nnet3> ") {
        std::cerr << "Kaldi Nnet3 text file should start with `<Nnet3>` and end with `</Nnet3>` tag." << std::endl;
        exit(1);
    }
    else {
        lines.erase(lines.begin());
        lines.pop_back();
        lines_num -= 2;
    }
    
    unsigned long i = -1;
    unsigned long start = 0;
    std::vector<std::string> line_split;
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> nodes;
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<float>>> nodes_matrices;
    
    for (auto line : lines) {
        i++;
        if (line.length() > 0) {
            line_split = split(line, ' ');
            if (line_split[0] == "<NumComponents>") {
                start = i + 1;
                break;
            }
            else {
                std::string type = line_split[0];
                line_split.erase(line_split.begin());
                std::unordered_map<std::string, std::string> node = ParseNodeAttributes(line_split, type);
//                for (auto x : node)
//                    std::cout << x.first << ": " << x.second << std::endl;
//                std::cout << std::endl << std::endl;
                std::string node_name = node["name"];
                
                // add the current node as output of its input node (both ways reference)
                if (is_in(node, "input")) {
                    std::string input_node_name = node["input"];
                    assert(nodes.find(input_node_name) != nodes.end());
                    nodes[input_node_name]["output"] = node_name;
                }
                nodes.insert({node_name, node});
            }
        }
    }
    
    lines.erase(lines.begin(), lines.begin() + start);
    std::string component_name(""), matrix_key("");
    std::unordered_map<std::string, std::string> component_attrs, component_attrs_part;
    std::unordered_map<std::string, std::vector<float>> matrix_attrs;
    std::vector<float> matrix;
    bool end = false;
    bool matrix_end = false;
    
    for (auto line : lines) {
        if (line.length() > 0) {
            line_split = split(line, ' ');
            if (line_split[0] == "<ComponentName>") {
                component_name = line_split[1];
                line_split.erase(line_split.begin(), line_split.begin() + 2);
                boost::replace_all(line_split[0], "<", "");
                boost::replace_all(line_split[0], ">", "");
                component_attrs["type"] = line_split[0];
                line_split.erase(line_split.begin());
            }
            if (component_name.length() > 0) {
                if (matrix_key.length() > 0) {
                    ParseFloatsLine(matrix, line_split, matrix_end);
                    if (matrix_end) {
                        matrix_attrs[matrix_key] = matrix;
                        matrix_key = "";
                        matrix.clear();
                    }
                }
                else {
                    component_attrs_part = ParseComponentAttributes(line_split, matrix_key, end, matrix_attrs);
                    // update components dictionary
                    for (auto x : component_attrs_part)
                        component_attrs[x.first] = x.second;
                    if (end) {
                        for (auto x : component_attrs)
                            nodes[component_name][x.first] = x.second;
                        for (auto x : matrix_attrs)
                            nodes_matrices[component_name][x.first] = x.second;
                        
                        // clear variables
                        component_name = "";
                        component_attrs.clear();
                        component_attrs_part.clear();
                        matrix_attrs.clear();
                    }
                }
            }
        }
    }
    
    unsigned int num_input_nodes = 0;
    std::unordered_map<std::string, std::string> input_node;
    std::unordered_map<std::string, std::vector<float>> input_node_matrices;
    
    for (auto node : nodes) {
        if (node.first == "input") {
            num_input_nodes++;
            input_node = node.second;
        }
//        std::cout << node.first << std::endl;
//        for (auto x : node.second) {
//            std::cout << "  " << x.first << ": " << x.second << std::endl;
//        }
//        for (auto x : nodes_matrices[node.first]) {
//            std::cout << "  " << x.first << ": " << x.second.size() << std::endl;
//        }
    }
    assert(num_input_nodes == 1);
    
    // iterate over layers and connect them starting from input
    while (input_node.find("output") != input_node.end()) {
        input_node = nodes[input_node["output"]];
        input_node_matrices = nodes_matrices[input_node["component"]];
        InitLayersFromNode(input_node, input_node_matrices);
    }
    std::cout << "Parsing of neural net config done." << std::endl;
}


std::unordered_map<std::string, std::string> NNet::ParseNodeAttributes(std::vector<std::string> attributes, std::string type) {
    std::unordered_map<std::string, std::string> output = {{"type", type}};
    std::vector<std::string> values;
    std::vector<std::string> offsets;
    std::string key(""), value("");
    for (auto attribute : attributes) {
        if (attribute.find("=") != std::string::npos) {
            if (key.length() > 0) {
                std::string values_string = join(values);
                offsets.clear();
                std::string name = ParseNodeAttributeValue(values_string, offsets);
                output.insert({key, name});
                if (offsets.size() > 0)
                    output.insert({"stacking", boost::join(offsets, " ")});
            }
            key = split(attribute, '=')[0];
            value = split(attribute, '=')[1];
            values.clear();
            values.push_back(value);
        }
        else {
            values.push_back(attribute);
        }
    }
    if (key.length() > 0) {
        std::string values_string = join(values);
        offsets.clear();
        std::string name = ParseNodeAttributeValue(values_string, offsets);
        output.insert({key, name});
        if (offsets.size() > 0)
            output.insert({"stacking", boost::join(offsets, " ")});
    }
    return output;
}


std::string NNet::ParseNodeAttributeValue(std::string value, std::vector<std::string> &offsets) {
    std::string name;
    std::vector<std::string> names;
    std::vector<std::string> splitted;
    if (startswith(value, "Append(")) {
        // remove 7 characters from start and one character from end
        value.erase(0, 7);
        value.erase(value.end() - 1, value.end());
        splitted.clear();
        for (auto item : boost::split(splitted, value, boost::is_any_of(","))) {
            boost::replace_all(item, " ", "");
            if (startswith(item, "Offset(")) {
                item.erase(0, 7);
                name = item;
            }
            else {
                if (name.length() > 0) {
                    names.push_back(name);
                    boost::replace_all(item, ")", "");
                    offsets.push_back(item);
                    name = "";
                }
                else {
                    names.push_back(item);
                    offsets.push_back("0");
                }
            }
        }
        std::set<std::string> names_set(names.begin(), names.end());
        if (names_set.size() == 1)
            return names[0];
    }
    if (startswith(value, "Round(")) {
        value.erase(0, 6);
        value.erase(value.end() - 1, value.end());
        splitted.clear();
        boost::split(splitted, value, boost::is_any_of(","));
        offsets.clear();
        return splitted[0];
    }
    return value;
}


std::unordered_map<std::string, std::string> NNet::ParseComponentAttributes(                                                                    std::vector<std::string> line_split, std::string &matrix_key, bool &end, std::unordered_map<std::string, std::vector<float>> &matrix_attrs) {
    std::unordered_map<std::string, std::string> output;
    std::string value;
    end = false;
    bool tmp_end = false;
    
    while (line_split.size() > 0) {
        boost::replace_all(line_split[0], "<", "");
        boost::replace_all(line_split[0], ">", "");
        matrix_key = line_split[0];
        line_split.erase(line_split.begin());
        if (matrix_key[0] == '/') {
            end = true;
        }
        else {
            value = line_split[0];
            line_split.erase(line_split.begin());
            if (value == "[") {
                if (line_split.size() > 0) {
                    ParseFloatsLine(matrix_attrs[matrix_key], line_split, tmp_end);
                    matrix_key = "";
                }
                return output;
            }
            else {
                output[matrix_key] = value;
            }
        }
    }
    matrix_key = "";
    return output;
}


void NNet::ParseFloatsLine(std::vector<float> &matrix, std::vector<std::string> line_split, bool &matrix_end) {
    if (line_split[line_split.size() - 1] == "]") {
        matrix_end = true;
        line_split.pop_back();
    }
    else {
        matrix_end = false;
    }
    for (auto x : line_split)
        matrix.push_back(std::stof(x));
}


void NNet::InitLayersFromNode(std::unordered_map<std::string, std::string> &node_attrs, std::unordered_map<std::string, std::vector<float>> &matrix_attrs) {
    std::string name = node_attrs["component"];
    std::string type = node_attrs["type"];
    bool is_initialized = false;
    
    if (type == "NaturalGradientAffineComponent") {
        if (node_attrs.find("stacking") != node_attrs.end()) {
            m_layers.push_back(new StackingLayer(name, str2ints(node_attrs["stacking"])));
            m_layers_types.push_back("NaturalGradientAffineComponent StackingLayer");
        }
        m_layers.push_back(new DenseLayer(name, m_context, matrix_attrs["LinearParams"], matrix_attrs["BiasParams"]));
        m_layers_types.push_back("NaturalGradientAffineComponent DenseLayer");
        is_initialized = true;
    }
    if (type == "RectifiedLinearComponent") {
        m_layers.push_back(new ReLULayer(name));
        m_layers_types.push_back("RectifiedLinearComponent");
        is_initialized = true;
    }
    if (type == "BatchNormComponent") {
        m_layers.push_back(new BatchNormLayer(name, m_context, matrix_attrs["StatsMean"], matrix_attrs["StatsVar"], std::stof(node_attrs["Epsilon"])));
        m_layers_types.push_back("BatchNormComponent");
        is_initialized = true;
    }
    if (type == "StatisticsExtractionComponent") {
        m_layers.push_back(new StatisticsExtractionLayer(name, node_attrs["IncludeVarinance"] == "T"));
        m_layers_types.push_back("StatisticsExtractionComponent");
        is_initialized = true;
    }
    if (type == "StatisticsPoolingComponent") {
        m_layers.push_back(new StatisticsPoolingLayer(name, std::stoi(node_attrs["InputDim"]), node_attrs["OutputStddevs"] == "T", std::stoi(node_attrs["LeftContext"]), std::stoi(node_attrs["RightContext"]), std::stof(node_attrs["VarianceFloor"])));
        m_layers_types.push_back("StatisticsPoolingComponent");
        is_initialized = true;
    }
    if (type == "output-node") {
        is_initialized = true;
        m_layers_types.push_back("output-node");
    }
    
    assert(is_initialized);
}


std::vector<float> NNet::forward(std::string fea_path, cl_device_id device, cl_context context, cl_command_queue queue, unsigned long &rows, unsigned long &cols) {
    std::cout << "Executing forward pass of neural net." << std::endl;
    cl_int err;
    
    std::vector<float> features = loadtxt(fea_path, rows, cols);
    cl_mem features_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * features.size(), &features[0], &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    cl_mem input = features_buffer;
    cl_mem output;

    std::string type;
    unsigned long fea_rows, fea_cols;
    fea_rows = rows;
    fea_cols = cols;
    
    for (unsigned int j = 0; j < 10; j ++) {
        input = features_buffer;
        rows = fea_rows;
        cols = fea_cols;
//        std::cout << features.size() << std::endl;
    for (unsigned int i = 0; i < m_layers.size(); i++) {
        type = m_layers_types[i];
        std::cout << "Processing layer " << i << " with type: " << type << std::endl;
        if (startswith(type, "NaturalGradientAffineComponent")) {
            if (type == "NaturalGradientAffineComponent StackingLayer") {
                StackingLayer *layer = dynamic_cast<StackingLayer*>(m_layers[i]);
                output = layer->forward(input, rows, cols, device, context, queue);
//                savetxt("/tmp/cpp_layer_" + std::to_string(i) + ".txt", enqueue_buffer(queue, output, rows, cols), rows, cols);
                input = output;
                i++;
                std::cout << "Processing layer " << i << " with type: " << m_layers_types[i] << std::endl;
            }
            DenseLayer *layer2 = dynamic_cast<DenseLayer*>(m_layers[i]);
            output = layer2->forward(input, rows, cols, device, context, queue);
        }
        else if (type == "RectifiedLinearComponent") {
            ReLULayer *relu_layer = dynamic_cast<ReLULayer*>(m_layers[i]);
            output = relu_layer->forward(input, rows, cols, device, context, queue);
        }
        else if (type == "BatchNormComponent") {
            BatchNormLayer *batchnorm_layer = dynamic_cast<BatchNormLayer*>(m_layers[i]);
            output = batchnorm_layer->forward(input, rows, cols, device, context, queue);
        }
        else if (type == "StatisticsExtractionComponent") {
            StatisticsExtractionLayer *statistics_extraction_layer = dynamic_cast<StatisticsExtractionLayer*>(m_layers[i]);
            output = statistics_extraction_layer->forward(input, rows, cols, device, context, queue);
        }
        else if (type == "StatisticsPoolingComponent") {
            StatisticsPoolingLayer *statistics_extraction_layer = dynamic_cast<StatisticsPoolingLayer*>(m_layers[i]);
            output = statistics_extraction_layer->forward(input, rows, cols, device, context, queue);
        }
        else {
            std::cerr << "Unexpected type of node " << type << "." << std::endl;
            exit(1);
        }
//        savetxt("/tmp/cpp_layer_" + std::to_string(i) + ".txt", enqueue_buffer(queue, output, rows, cols), rows, cols);
        input = output;
    }
        clFinish(queue);
    }
    
    std::vector<float> output_vec = enqueue_buffer(queue, output, rows, cols);
    return output_vec;
}
