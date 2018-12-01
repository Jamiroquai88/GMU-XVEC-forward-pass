//
//  utils.hpp
//  GMU-XVEC
//
//  Created by Jan Profant on 28/11/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#ifndef utils_h
#define utils_h

#include <algorithm>
#include <string>
#include <sstream>
#include <vector>
#include <iterator>


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


std::string join(std::vector<std::string> strings, std::string delim=" ") {
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



#endif /* utils_h */
