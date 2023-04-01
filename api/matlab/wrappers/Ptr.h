#ifndef ARRUS_API_MATLAB_WRAPPERS_ARRUS_MEXOBJECTWRAPPER_H
#define ARRUS_API_MATLAB_WRAPPERS_ARRUS_MEXOBJECTWRAPPER_H

#include <memory>
#include <unordered_map>
#include <utility>

#include "api/matlab/wrappers/MexContext.h"
#include "api/matlab/wrappers/common.h"

namespace arrus::matlab {

class Ptr {
public:
    explicit Ptr(std::shared_ptr<MexContext> ctx) : ctx(std::move(ctx)) {}

    explicit Ptr(MatlabObjectHandle handle, std::shared_ptr<MexContext> ctx) : ctx(std::move(ctx)) {}

    virtual ~Ptr() = default;

    virtual MatlabOutputArgs call(const MatlabMethodId &methodId, MatlabInputArgs &inputs) {
        return methods.at(methodId)(inputs);
    }

protected:
    using MexObjectMethod = std::function<MatlabOutputArgs(MatlabInputArgs &)>;

    void addMethod(const MatlabClassId &methodId, const MexObjectMethod &method) {
        methods.emplace(methodId, method);
    }

    std::string className{};
    MatlabObjectHandle handle{0};
    std::shared_ptr<MexContext> ctx;
    std::unordered_map<MatlabClassId, MexObjectMethod> methods;
};
}// namespace arrus::matlab

#endif//ARRUS_API_MATLAB_WRAPPERS_ARRUS_MEXOBJECTWRAPPER_H
