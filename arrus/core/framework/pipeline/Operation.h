#ifndef CPP_EXAMPLE_IMAGING_OPERATION_H
#define CPP_EXAMPLE_IMAGING_OPERATION_H

#include <string>
#include <utility>
#include <unordered_map>
#include "NdArray.h"

namespace arrus_example_imaging {

#define OPERATION_CLASS_ID(Type) #Type

class OpParameters {
public:
    using Container = std::unordered_map<std::string, NdArray>;

    OpParameters() = default;

    explicit OpParameters(Container params)
        : params(std::move(params)) {}

    const NdArray &getArray(const std::string &key) const {
        try {
            return params.at(key);
        }
        catch(const std::out_of_range &e) {
            throw std::runtime_error("There is not Op parameter with key: " + key);
        }
    }
private:
    Container params;
};

class Operation {
public:
    using OpClassId = std::string;
    using OpId = std::string;

    Operation() = default;

    Operation(OpClassId classId, OpParameters params)
        : classId(std::move(classId)), params(std::move(params)) {}

    OpClassId getOpClassId() const { return classId; }

    const OpParameters &getParams() const { return params; }

private:
    OpClassId classId;
    OpParameters params;
};

class OperationBuilder {

public:
    OperationBuilder() = default;

    OperationBuilder& setClassId(Operation::OpClassId id) {
        this->classId = std::move(id);
        return *this;
    }

    OperationBuilder& addParam(const std::string& key, const NdArray &arr) {
        this->params[key] = arr;
        return *this;
    }

    Operation build() {
        return Operation{classId, OpParameters{params}};
    }

private:
    Operation::OpClassId classId;
    OpParameters::Container params;
};

}// namespace arrus_example_imaging

#endif//CPP_EXAMPLE_IMAGING_OPERATION_H
