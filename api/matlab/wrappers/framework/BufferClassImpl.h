#ifndef API_MATLAB_WRAPPERS_FRAMEWORK_BUFFERCLASSIMPL_H
#define API_MATLAB_WRAPPERS_FRAMEWORK_BUFFERCLASSIMPL_H

#include <ostream>
#include <string>
#include <utility>

#include "api/matlab/wrappers/ClassObjectManager.h"
#include "api/matlab/wrappers/MatlabStdoutBuffer.h"
#include "api/matlab/wrappers/Ptr.h"
#include "api/matlab/wrappers/asserts.h"
#include "api/matlab/wrappers/common.h"
#include "api/matlab/wrappers/convert.h"
#include "api/matlab/wrappers/framework/LockBasedBuffer.h"
#include "arrus/common/format.h"
#include "arrus/core/api/arrus.h"

namespace arrus::matlab::framework {

using namespace ::arrus::matlab;
using namespace ::arrus::framework;
using namespace ::arrus::matlab::converters;

class BufferClassImpl : public ClassObjectManager<LockBasedBuffer> {
public:
    inline static const std::string CLASS_NAME = "arrus.framework.Buffer";

    explicit BufferClassImpl(const std::shared_ptr<MexContext> &ctx) : ClassObjectManager(ctx, CLASS_NAME) {
        ARRUS_MATLAB_ADD_METHOD("front", front);
        ARRUS_MATLAB_ADD_METHOD("back", back);
    }

    MatlabObjectHandle create(std::shared_ptr<MexContext> ctx, ::arrus::matlab::MatlabInputArgs &args) override {
        ARRUS_MATLAB_REQUIRES_N_PARAMETERS_CLASS_METHOD(args, 1, CLASS_NAME, "constructor");
        auto arg = args[0];
        ARRUS_MATLAB_REQUIRES_SCALAR(arg);
        ARRUS_MATLAB_REQUIRES_TYPE(arg, ::matlab::data::ArrayType::UINT64);
        size_t value = arg[0];// Pointer to the buffer returned by Session::upload
        std::shared_ptr<DataBuffer> dataBuffer(reinterpret_cast<DataBuffer *>(value));
        return insert(std::make_unique<LockBasedBuffer>(dataBuffer));
    }
    void remove(const MatlabObjectHandle handle) override { ClassObjectManager::remove(handle); }

    void front(MatlabObjectHandle obj, MatlabOutputArgs &outputs, MatlabInputArgs &inputs) {
        auto *array = get(obj);
        auto element = array->front();
        if (element.has_value()) {
            // The buffer is still open.
            auto ptr = reinterpret_cast<size_t>(element->get());
            outputs[0] = ARRUS_MATLAB_GET_MATLAB_SCALAR(ctx, size_t, ptr);
        } else {
            throw ::arrus::IllegalStateException("Buffer was already closed.");
        }
    }

    void back(MatlabObjectHandle obj, MatlabOutputArgs &outputs, MatlabInputArgs &inputs) {
        auto *array = get(obj);
        auto element = array->back();
        if (element.has_value()) {
            // The buffer is still open.
            auto ptr = reinterpret_cast<size_t>(element->get());
            outputs[0] = ARRUS_MATLAB_GET_MATLAB_SCALAR(ctx, size_t, ptr);
        } else {
            throw ::arrus::IllegalStateException("Buffer was already closed.");
        }
    }
};

}// namespace arrus::matlab::framework

#endif//API_MATLAB_WRAPPERS_FRAMEWORK_BUFFERCLASSIMPL_H
