#ifndef ARRUS_API_MATLAB_WRAPPERS_SESSION_SESSIONOBJECTWRAPPER_H
#define ARRUS_API_MATLAB_WRAPPERS_SESSION_SESSIONOBJECTWRAPPER_H

#include <utility>
#include <string>

#include "arrus/api/matlab/wrappers/MexObjectWrapper.h"
#include "arrus/api/matlab/wrappers/DefaultMexObjectManager.h"
#include "arrus/api/matlab/wrappers/asserts.h"
#include "arrus/api/matlab/wrappers/devices/convertDeviceId.h"

#include "arrus/core/api/session/Session.h"
#include "arrus/core/api/devices/DeviceId.h"
#include "arrus/common/format.h"


namespace arrus::matlab {

class SessionWrapper : public MexObjectWrapper {
public:

    explicit SessionWrapper(
            MexContext::SharedHandle ctx,
            const MexMethodArgs &args)
            : MexObjectWrapper(std::move(ctx)) {

        ARRUS_MATLAB_ADD_METHOD(this, "getDevice", SessionWrapper::getDevice);
    }

    ~SessionWrapper() override {
        ctx->logInfo("Destructor");
    }

    MexMethodReturnType getDevice(MexMethodArgs &inputs) {
        ARRUS_MATLAB_REQUIRES_N_PARAMETERS(inputs, 1, "getDevice");

        DeviceId deviceId = convertToDeviceId(ctx, inputs[0]);
        ctx->logInfo(arrus::format("Got DeviceId: {}", deviceId.toString()));
        return ctx->getArrayFactory().createCellArray({1, 1},
            ctx->getArrayFactory().createArray<double>({2, 2}, {1., 2., 3., 4.}));
    }

};

class SessionWrapperManager : public DefaultMexObjectManager<SessionWrapper> {
    using DefaultMexObjectManager::DefaultMexObjectManager;
};

}

#endif
