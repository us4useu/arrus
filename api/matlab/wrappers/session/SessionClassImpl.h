#ifndef ARRUS_API_MATLAB_WRAPPERS_SESSION_SESSIONOBJECTWRAPPER_H
#define ARRUS_API_MATLAB_WRAPPERS_SESSION_SESSIONOBJECTWRAPPER_H

#include <ostream>
#include <string>
#include <utility>

#include "api/matlab/wrappers/MatlabClassObjectManager.h"
#include "api/matlab/wrappers/MatlabStdoutBuffer.h"
#include "api/matlab/wrappers/Ptr.h"
#include "api/matlab/wrappers/asserts.h"

#include "arrus/common/format.h"
#include "arrus/core/api/devices/DeviceId.h"
#include "arrus/core/api/session/Session.h"
#include "arrus/core/api/session/SessionSettings.h"

namespace arrus::matlab {

class SessionClassImpl : public MatlabClassObjectManager<::arrus::session::Session> {
public:
    inline static const std::string CLASS_NAME = "arrus.ops.session.Session";

    SessionClassImpl(const std::shared_ptr<MexContext> &ctx): MatlabClassObjectManager(ctx, CLASS_NAME) {
        // Fix me?
        ARRUS_MATLAB_ADD_METHOD("getDevice", getDevice);
    }

    MatlabObjectHandle create(std::shared_ptr<MexContext> ctx, MatlabMethodArgs &args) override {
        ARRUS_MATLAB_REQUIRES_N_PARAMETERS_CLASS_METHOD(args, 1, CLASS_NAME, "constructor");
        auto arg = args[0];
        ARRUS_MATLAB_REQUIRES_SCALAR(arg, std::string("session constructor parameters"));
        // TODO an ObjectArray will be accepted in the future
        ARRUS_MATLAB_REQUIRES_TYPE(arg, ::matlab::data::ArrayType::MATLAB_STRING,
                                   std::string("session constructor parameters"));
        std::string cfgPath = arg[0];
        std::unique_ptr<ValueType> sess = ::arrus::session::createSession(cfgPath);
        return insert(std::move(sess));
    }

    MatlabMethodReturnType getDevice(MatlabObjectHandle obj, MatlabMethodArgs &inputs) {
        ARRUS_MATLAB_REQUIRES_N_PARAMETERS(inputs, 1, "getDevice");
    }

};


}// namespace arrus::matlab

#endif
