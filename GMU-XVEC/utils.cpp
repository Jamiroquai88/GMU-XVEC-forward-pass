//
//  utils.cpp
//  GMU-XVEC
//
//  Created by Jan Profant on 08/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#include <stdio.h>
#include "utils.hpp"

#define MAC

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif


std::vector<std::string> split(const std::string &s, char delim) {
    std::stringstream ss(s);
    std::string item;
    std::vector<std::string> result;
    while (std::getline(ss, item, delim)) {
        if (item.length() > 0)
            result.push_back(item);
    }
    return result;
}


bool startswith(const std::string& input, const std::string& match)
{
    return input.size() >= match.size() && equal(match.begin(), match.end(), input.begin());
}


bool is_in(const std::vector<std::string>& input, const std::string element) {
    for (auto x : input) {
        if (x == element)
            return true;
    }
    return false;
}


bool is_in(std::unordered_map<std::string, std::string>& input, const std::string element) {
    for (auto& it: input) {
        if (it.first == element)
            return true;
    }
    return false;
}


std::string join(std::vector<std::string> strings, std::string delim) {
    std::string result(strings[0]);
    for(auto it = strings.begin(); it != strings.end(); ++it) {
        if (it != strings.begin()) {
            if (std::next(it) != strings.end())
                result += *it + delim;
            else
                result += *it;
        }
    }
    return result;
}


std::vector<int> str2ints(std::string str) {
    std::vector<int> output;
    for (auto x : split(str, ' '))
        output.push_back(std::stof(x));
    return output;
}


void transpose(float *src, float *dst, const unsigned long N, const unsigned long M) {
    for(int n = 0; n<N*M; n++) {
        int i = n/N;
        int j = n%N;
        dst[n] = src[M*j + i];
    }
}


std::vector<float> loadtxt(std::string fea_path, unsigned long &num_samples, unsigned long &num_dims) {
    std::string line;
    std::vector<std::string> lines;
    std::ifstream infile(fea_path);
    if (!infile.good()) {
        std::cerr << "Path to feature file does not exist: " << fea_path << std::endl;
        exit(1);
    }
    
    while (std::getline(infile, line)) {
        lines.push_back(line);
    }
    
    unsigned long lines_num = lines.size();
    std::vector<float> output;
    std::vector<std::string> splitted_line;
    unsigned long current_num_dims;
    unsigned long old_num_dims = 0;
    
    for (auto line : lines) {
        splitted_line = split(line, ' ');
        current_num_dims = splitted_line.size();
        if (old_num_dims == 0)
            old_num_dims = current_num_dims;
        else {
            if (old_num_dims != current_num_dims) {
                std::cerr << "Incorrect format of input features in file: " << fea_path << std::endl;
                exit(1);
            }
        }
        for (auto x : splitted_line)
            output.push_back(std::stof(x));
    }
    num_samples = lines_num;
    num_dims = old_num_dims;
    
    //    std::vector<float> output_transposed(output.size());
    //    transpose(&output[0], &output_transposed[0], num_samples, num_dims);
    //    output.clear();
    //    return output_transposed;
    return output;
}


void savetxt(std::string output_fname, std::vector<float> matrix, unsigned long rows, unsigned long cols) {
    std::ofstream myfile(output_fname);
    if (myfile.is_open()) {
        for (unsigned long x = 0; x < rows; x++) {
            for (unsigned long y = 0; y < cols; y++) {
                myfile << matrix[x * cols + y] << " ";
            }
            myfile << "\n";
        }
    }
}


bool allclose(std::vector<float> v1, std::vector<float> v2, float eps) {
    if (v1.size() == v2.size()) {
        for (unsigned long i = 0; i < v1.size(); i++) {
            if (abs(v1[i] - v2[i]) > eps) {
                return false;
            }
        }
    }
    else {
        return false;
    }
    return true;
}

