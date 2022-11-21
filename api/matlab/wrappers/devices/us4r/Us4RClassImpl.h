#ifndef API_MATLAB_WRAPPERS_DEVICES_US4R_US4RCLASSIMPL_H
#define API_MATLAB_WRAPPERS_DEVICES_US4R_US4RCLASSIMPL_H

#include <ostream>
#include <string>
#include <utility>

#include "api/matlab/wrappers/ClassObjectWrapper.h"
#include "api/matlab/wrappers/Ptr.h"
#include "api/matlab/wrappers/asserts.h"
#include "api/matlab/wrappers/common.h"
#include "api/matlab/wrappers/convert.h"
#include "api/matlab/wrappers/devices/probe/ProbeModelConverter.h"
#include "arrus/common/format.h"
#include "arrus/core/api/arrus.h"

namespace arrus::matlab::devices {

using namespace arrus::matlab;
using namespace arrus::devices;

class Us4RClassImpl : public ClassObjectWrapper<Us4R> {
public:
    inline static const std::string CLASS_NAME = "arrus.devices.us4r.Us4R";

    explicit Us4RClassImpl(const std::shared_ptr<MexContext> &ctx) : ClassObjectWrapper(ctx, CLASS_NAME) {
        ARRUS_MATLAB_ADD_METHOD("setVoltage", setVoltage);
        ARRUS_MATLAB_ADD_METHOD("getSamplingFrequency", getSamplingFrequency);
        ARRUS_MATLAB_ADD_METHOD("getProbeModel", getProbeModel);
        ARRUS_MATLAB_ADD_METHOD("getChannelsMask", getChannelsMask);
    }

    void setVoltage(MatlabObjectHandle obj, MatlabOutputArgs &outputs, MatlabInputArgs &inputs) {
        ::arrus::Voltage value = inputs[0][0];
        ctx->logInfo(format("Us4R: setting voltage {}", value));
        get(obj)->setVoltage(value);
    }

    void getSamplingFrequency(MatlabObjectHandle obj, MatlabOutputArgs &outputs, MatlabInputArgs &inputs) {
        float fs = get(obj)->getSamplingFrequency();
        outputs[0] = ARRUS_MATLAB_GET_MATLAB_SCALAR(ctx, float, fs);
    }

    void getProbeModel(MatlabObjectHandle obj, MatlabOutputArgs &outputs, MatlabInputArgs &inputs) {
        auto &model = get(obj)->getProbe(0)->getModel();
        outputs[0] = ::arrus::matlab::devices::probe::ProbeModelConverter::from(ctx, model).toMatlab();
    }

    void getChannelsMask(MatlabObjectHandle obj, MatlabOutputArgs &outputs, MatlabInputArgs &inputs) {
        auto channelsMask = get(obj)->getChannelsMask();
        outputs[0] = ARRUS_MATLAB_GET_MATLAB_VECTOR(ctx, unsigned short, channelsMask);
    }

};

}

#endif//API_MATLAB_WRAPPERS_DEVICES_US4R_US4RCLASSIMPL_H
