#ifndef ARRUS_CORE_API_FRAMEWORK_PIPELINE_STEP_H
#define ARRUS_CORE_API_FRAMEWORK_PIPELINE_STEP_H

#include <string>
#include <unordered_map>
#include <utility>

namespace arrus::framework::pipeline {

class Step {
public:
    using Handle = std::unique_ptr<Step>;
    using ClassId = std::string;
    using ParamKey = std::string;
    using ParamValue = NdArray;
    using Params = std::unordered_map<ParamKey, ParamValue>;

    Step(ClassId classId, Params params)
        : classId(std::move(classId)), params(std::move(params)) {}

private:
    ClassId classId;
    Params params;
};


class StageBuilder {
public:
    StageBuilder() = default;

    void setClass(Step::ClassId cls) {
        this->classId = std::move(cls);
    }

    void addParam(const Step::ParamKey &key, Step::ParamValue value) {
        params[key] = std::move(value);
    }

private:
    Step::ClassId classId;
    Step::Params params;
};


}

#endif//ARRUS_CORE_API_FRAMEWORK_PIPELINE_STEP_H
