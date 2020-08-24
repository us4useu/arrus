#ifndef ARRUS_API_MATLAB_WRAPPERS_ARRUS_MEXOBJECTWRAPPER_H
#define ARRUS_API_MATLAB_WRAPPERS_ARRUS_MEXOBJECTWRAPPER_H

#include <unordered_map>
#include <memory>
#include <utility>

#include "api/matlab/wrappers/common.h"
#include "api/matlab/wrappers/MexContext.h"

namespace arrus::matlab {

class MexObjectWrapper {
public:
    explicit MexObjectWrapper(std::shared_ptr<MexContext> ctx)
            : ctx(std::move(ctx)) {}

    virtual ~MexObjectWrapper() = default;

    virtual MexMethodReturnType
    call(const MexObjectMethodId &methodId, const MexMethodArgs &inputs) {
        return methods.at(methodId)(inputs);
    }

protected:
    using MexObjectMethod = std::function<MexMethodReturnType(
            const MexMethodArgs &)>;

    void
    addMethod(const MexObjectClassId &methodId, const MexObjectMethod &method) {
        methods.emplace(methodId, method);
    }

    std::shared_ptr<MexContext> ctx;
    std::unordered_map<MexObjectClassId, MexObjectMethod> methods;
};
}


#endif //ARRUS_API_MATLAB_WRAPPERS_ARRUS_MEXOBJECTWRAPPER_H
