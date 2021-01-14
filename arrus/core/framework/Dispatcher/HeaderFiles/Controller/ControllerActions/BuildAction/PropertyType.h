#pragma once

#include <regex>
#include <boost/property_tree/ptree.hpp>
#include "Model/VariableAnyValue.h"
#include <unordered_map>

class PropertyType {
public:
    enum class InnerType {
        BOOL, INT, FLOAT, STRING, ARRAY, MAP, MAP_OR_ARRAY_OR_STRING, POINTER
    };

    PropertyType();

    PropertyType(const VariableAnyValue &val);

    PropertyType(const boost::property_tree::ptree &prop);

    PropertyType(InnerType type);

    bool operator==(const PropertyType &type) const;

    bool operator!=(const PropertyType &type) const;

    std::string toString();

private:
    std::regex pointer_regex, int_regex, float_regex, bool_regex;
    InnerType innerType;

    InnerType getInnerType(const boost::property_tree::ptree &prop);

    void initiateRegexes();
};