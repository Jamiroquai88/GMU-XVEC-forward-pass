//
//  nnet.cpp
//  GMU-XVEC
//
//  Created by Jan Profant on 28/11/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#include "nnet.hpp"
#include "utils.hpp"

#include <fstream>
#include <vector>
#include <unordered_map>


NNet::NNet(std::string nnet_path) {
    std::cout << nnet_path;
    std::string line;
    std::vector<std::string> lines;
    std::ifstream infile(nnet_path);
    
    while (std::getline(infile, line)) {
        lines.push_back(line);
    }
    
    unsigned long lines_num = lines.size();
    
    std::cout << "|" << lines[0] << "|";
    std::cout << "|" << lines[lines_num - 1] << "|";
    
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
    for (auto line : lines) {
        i++;
        line_split = split(line, ' ');
        if (line_split[0] == "<NumComponents>") {
            start = i + 1;
            break;
        }
        else {
            std::string type = line_split[0];
            line_split.erase(line_split.begin());
            std::unordered_map<std::string, std::string> node = ParseNodeAttributes(line_split, type);
            std::string node_name = node["name"];
            
        }
    }
    
    
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
                    output.insert({"stacking", join(offsets)});
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
            output.insert({"stacking", join(offsets)});
    }
    return output;
}


std::string NNet::ParseNodeAttributeValue(std::string value, std::vector<std::string> offsets) {
    if (startswith(value, "Append")) {
        // TODO
    }
    if (startswith(value, "Round")) {
        // TODO
    }
    return value;
}
