#ifndef ARRUS_API_MATLAB_WRAPPERS_ARRUS_MEXOBJECTWRAPPER_H
#define ARRUS_API_MATLAB_WRAPPERS_ARRUS_MEXOBJECTWRAPPER_H

#include <memory>
#include <unordered_map>
#include <utility>

#include "api/matlab/wrappers/MexContext.h"
#include "api/matlab/wrappers/common.h"

namespace arrus::matlab {

// TODO MexObjectWrapper should store pointer to the underlying object
// TODO on destruction, SessionWrapper should destroy underlying object
// TODO 'methods' should not be an object property, but the class

class MexObjectWrapper {
public:
    explicit MexObjectWrapper(MexObjectHandle handle, std::shared_ptr<MexContext> ctx) : ctx(std::move(ctx)) {}

    virtual ~MexObjectWrapper() = default;

    virtual MexMethodReturnType call(const MexObjectMethodId &methodId, MexMethodArgs &inputs) {
        return methods.at(methodId)(inputs);
    }

protected:
    // Note: most of the MexRange methods (e.g. size) are not const, thus const
    // qualifier cannot be applied to the inputs.
    using MexObjectMethod = std::function<MexMethodReturnType(MexMethodArgs &)>;

    void addMethod(const MexObjectClassId &methodId, const MexObjectMethod &method) {
        methods.emplace(methodId, method);
    }
    MexObjectHandle handle;
    std::shared_ptr<MexContext> ctx;
    std::unordered_map<MexObjectClassId, MexObjectMethod> methods;
};

/**
 * A macro that adds method to given mex object wrapper.
 */
#define ARRUS_MATLAB_ADD_METHOD(obj, methodStr, method)                                                                \
    obj->addMethod(methodStr, std::bind(&method, obj, std::placeholders::_1))

}// namespace arrus::matlab

#endif//ARRUS_API_MATLAB_WRAPPERS_ARRUS_MEXOBJECTWRAPPER_H
