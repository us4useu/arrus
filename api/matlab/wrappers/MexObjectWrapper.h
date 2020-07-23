#ifndef ARRUS_API_MATLAB_WRAPPERS_ARRUS_MEXOBJECTWRAPPER_H
#define ARRUS_API_MATLAB_WRAPPERS_ARRUS_MEXOBJECTWRAPPER_H

#include <unordered_map>
#include <memory>
#include <utility>

#include "api/matlab/wrappers/common.h"
#include "api/matlab/wrappers/MexContext.h"

namespace arrus::matlab_wrappers {

class MexObjectWrapper {
public:
    explicit MexObjectWrapper(std::shared_ptr<MexContext> ctx)
            : ctx(std::move(ctx)) {}

    virtual void call(const MexObjectMethodId &, const MexMethodArgs &inputs,
         const MexMethodArgs &outputs) = 0;

protected:
    std::shared_ptr<MexContext> ctx;
};
}


#endif //ARRUS_API_MATLAB_WRAPPERS_ARRUS_MEXOBJECTWRAPPER_H
