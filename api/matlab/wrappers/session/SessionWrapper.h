#ifndef ARRUS_API_MATLAB_WRAPPERS_SESSION_SESSIONOBJECTWRAPPER_H
#define ARRUS_API_MATLAB_WRAPPERS_SESSION_SESSIONOBJECTWRAPPER_H

#include <ostream>
#include <string>
#include <utility>

#include "api/matlab/wrappers/DefaultMexObjectManager.h"
#include "api/matlab/wrappers/MatlabStdoutBuffer.h"
#include "api/matlab/wrappers/MexObjectWrapper.h"
#include "api/matlab/wrappers/asserts.h"
#include "api/matlab/wrappers/devices/convertDeviceId.h"
#include "api/matlab/wrappers/session/convertSessionSettings.h"

#include "arrus/common/format.h"
#include "arrus/core/api/devices/DeviceId.h"
#include "arrus/core/api/session/Session.h"
#include "arrus/core/api/session/SessionSettings.h"

namespace arrus::matlab {

class SessionWrapper : public MexObjectWrapper {
public:

    SessionWrapper(MexContext::SharedHandle ctx, MexMethodArgs &inputs) : MexObjectWrapper(std::move(ctx)) {
        // Callable methods declaration.
        ARRUS_MATLAB_ADD_METHOD(this, "getDevice", SessionWrapper::getDevice);
        // Read constructor parameters.
        //
        ARRUS_MATLAB_REQUIRES_N_PARAMETERS(inputs, 1, "constructor");
        // Two possible arguments.

        SessionSettings settings = convertToSessionSettings(this->ctx, inputs[0]);
        session = createSession(settings);
    }

    ~SessionWrapper() override = default;

    MexMethodReturnType getDevice(MexMethodArgs &inputs) {
        ARRUS_MATLAB_REQUIRES_N_PARAMETERS(inputs, 1, "getDevice");

        arrus::devices::DeviceId deviceId = convertToDeviceId(ctx, inputs[0]);
        // Create new mex object
        ctx->logInfo(arrus::format("Got DeviceId: {}", deviceId.toString()));
        return ctx->getArrayFactory().createCellArray(
            {1, 1}, ctx->getArrayFactory().createArray<double>({2, 2}, {1., 2., 3., 4.}));
    }

private:
    Session::Handle session;
};

class SessionWrapperManager : public DefaultMexObjectManager<SessionWrapper> {
    using DefaultMexObjectManager::DefaultMexObjectManager;
};

}// namespace arrus::matlab

#endif
