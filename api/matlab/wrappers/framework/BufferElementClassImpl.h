#ifndef API_MATLAB_WRAPPERS_FRAMEWORK_BUFFERELEMENTCLASSIMPL_H
#define API_MATLAB_WRAPPERS_FRAMEWORK_BUFFERELEMENTCLASSIMPL_H

#include <ostream>
#include <string>
#include <utility>

#include "api/matlab/wrappers/ClassObjectWrapper.h"
#include "api/matlab/wrappers/Ptr.h"
#include "api/matlab/wrappers/asserts.h"
#include "api/matlab/wrappers/common.h"
#include "api/matlab/wrappers/convert.h"
#include "arrus/common/format.h"
#include "arrus/core/api/arrus.h"

namespace arrus::matlab::framework {

using namespace ::arrus::matlab;
using namespace ::arrus::framework;

class BufferElementClassImpl : public ClassObjectWrapper<BufferElement> {
public:
    inline static const std::string CLASS_NAME = "arrus.framework.BufferElement";

    explicit BufferElementClassImpl(const std::shared_ptr<MexContext> &ctx) : ClassObjectWrapper(ctx, CLASS_NAME) {
        ARRUS_MATLAB_ADD_METHOD("eval", eval);
    }

    void eval(MatlabObjectHandle obj, MatlabOutputArgs &outputs, MatlabInputArgs &inputs) {
        auto *element = get(obj);
        outputs[0] = ctx->createArray(element->getData()); // Copy the data to a MATLAB array.
        element->release();// Release this buffer element.
    }
};

}// namespace arrus::matlab::framework

#endif//API_MATLAB_WRAPPERS_FRAMEWORK_BUFFERELEMENTCLASSIMPL_H
