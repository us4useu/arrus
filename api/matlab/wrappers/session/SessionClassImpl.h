#ifndef ARRUS_API_MATLAB_WRAPPERS_SESSION_SESSIONOBJECTWRAPPER_H
#define ARRUS_API_MATLAB_WRAPPERS_SESSION_SESSIONOBJECTWRAPPER_H

#include <ostream>
#include <string>
#include <utility>

#include "api/matlab/wrappers/ClassObjectManager.h"
#include "api/matlab/wrappers/MatlabStdoutBuffer.h"
#include "api/matlab/wrappers/Ptr.h"
#include "api/matlab/wrappers/asserts.h"

#include "arrus/common/format.h"
#include "arrus/core/api/devices/DeviceId.h"
#include "arrus/core/api/session/Session.h"
#include "arrus/core/api/session/SessionSettings.h"

namespace arrus::matlab {

using namespace ::arrus::matlab::converters;

class SessionClassImpl : public ClassObjectManager<::arrus::session::Session> {
public:
    inline static const std::string CLASS_NAME = "arrus.session.Session";

    explicit SessionClassImpl(const std::shared_ptr<MexContext> &ctx) : ClassObjectManager(ctx, CLASS_NAME) {
        ARRUS_MATLAB_ADD_METHOD("getDevice", getDevice);
        ARRUS_MATLAB_ADD_METHOD("upload", upload);
        ARRUS_MATLAB_ADD_METHOD("startScheme", startScheme);
        ARRUS_MATLAB_ADD_METHOD("stopScheme", stopScheme);
        ARRUS_MATLAB_ADD_METHOD("run", run);
        ARRUS_MATLAB_ADD_METHOD("close", close);
    }

    MatlabObjectHandle create(std::shared_ptr<MexContext> ctx, MatlabInputArgs &args) override {
        ARRUS_MATLAB_REQUIRES_N_PARAMETERS_CLASS_METHOD(args, 1, CLASS_NAME, "constructor");
        auto arg = args[0];
        ARRUS_MATLAB_REQUIRES_SCALAR(arg);
        ARRUS_MATLAB_REQUIRES_TYPE(arg, ::matlab::data::ArrayType::MATLAB_STRING);
        std::string cfgPath = arg[0];
        ctx->logInfo(format("Creating session using configuration file: {}", cfgPath));
        std::unique_ptr<ValueType> sess = ::arrus::session::createSession(cfgPath);
        return insert(std::move(sess));
    }
    void remove(const MatlabObjectHandle handle) override {
        ClassObjectManager::remove(handle);
    }

    void getDevice(MatlabObjectHandle obj, MatlabOutputArgs &outputs, MatlabInputArgs &inputs) {
        ARRUS_MATLAB_REQUIRES_N_PARAMETERS(inputs, 1, "getDevice");
        auto arg = inputs[0];
        ARRUS_MATLAB_REQUIRES_SCALAR(arg);
        ARRUS_MATLAB_REQUIRES_TYPE(arg, ::matlab::data::ArrayType::MATLAB_STRING);

        std::string deviceId = arg[0];

        // Reading information.
        ::arrus::session::Session *session = get(obj);
        auto *device = session->getDevice(deviceId);
        auto deviceAddr = reinterpret_cast<MatlabObjectHandle>(device);

        // Convert to some specific MATLAB type, based on the device type.
        switch (device->getDeviceId().getDeviceType()) {
        case devices::DeviceType::Us4R:
            // Us4R device.
            outputs[0] = ctx->createObject("arrus.devices.us4r.Us4R",
                                           {
                                               ARRUS_MATLAB_GET_MATLAB_SCALAR(ctx, uint64_t, deviceAddr),
                                           });
            break;
        default:
            // Some generic device.
            outputs[0] = ctx->createObject("arrus.devices.Device",
                                           {
                                               ARRUS_MATLAB_GET_MATLAB_SCALAR(ctx, uint64_t, deviceAddr),
                                           });
            break;
        }
    }

    void upload(MatlabObjectHandle obj, MatlabOutputArgs &outputs, MatlabInputArgs &inputs) {
        ARRUS_MATLAB_REQUIRES_N_PARAMETERS(inputs, 1, "upload");
        auto arg = inputs[0];
        ARRUS_MATLAB_REQUIRES_SCALAR(arg);
        ARRUS_MATLAB_REQUIRES_TYPE_NAMED(arg, ::matlab::data::ArrayType::VALUE_OBJECT, "scheme");

        ::arrus::ops::us4r::Scheme scheme = ::arrus::matlab::ops::us4r::SchemeConverter::from(
                                                ctx, ::arrus::matlab::converters::MatlabElementRef{inputs[0]})
                                                .toCore();
        auto *session = get(obj);
        auto uploadResult = session->upload(scheme);
        auto buffer = uploadResult.getBuffer();
        // Outputs
        outputs[0] = ARRUS_MATLAB_GET_MATLAB_SCALAR(ctx, MatlabObjectHandle, MatlabObjectHandle(buffer.get()));
        outputs[1] = ARRUS_MATLAB_GET_MATLAB_STRING(ctx, "test");
    }

    void startScheme(MatlabObjectHandle obj, MatlabOutputArgs &outputs, MatlabInputArgs &inputs) {
        auto *session = get(obj);
        ctx->logInfo("Starting scheme.");
        session->startScheme();
        ctx->logInfo("Scheme started.");
    }

    void stopScheme(MatlabObjectHandle obj, MatlabOutputArgs &outputs, MatlabInputArgs &inputs) {
        auto *session = get(obj);
        ctx->logInfo("Stopping scheme.");
        session->stopScheme();
        ctx->logInfo("Scheme stopped.");
    }

    void run(MatlabObjectHandle obj, MatlabOutputArgs &outputs, MatlabInputArgs &inputs) {
        auto *session = get(obj);
        session->run();
    }

    void close(MatlabObjectHandle obj, MatlabOutputArgs &outputs, MatlabInputArgs &inputs) {
        auto *session = get(obj);
        ctx->logInfo("Closing session.");
        session->close();
        ctx->logInfo("Session closed.");
    }

};

}// namespace arrus::matlab

#endif
