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
        ARRUS_MATLAB_ADD_METHOD("disableHV", disableHV);
        ARRUS_MATLAB_ADD_METHOD("getSamplingFrequency", getSamplingFrequency);
        ARRUS_MATLAB_ADD_METHOD("getProbeModel", getProbeModel);
        ARRUS_MATLAB_ADD_METHOD("getChannelsMask", getChannelsMask);
        ARRUS_MATLAB_ADD_METHOD("setTgcCurveValue", setTgcCurveValue);
        ARRUS_MATLAB_ADD_METHOD("setTgcCurveTimeValue", setTgcCurveTimeValue);
        ARRUS_MATLAB_ADD_METHOD("setVcatTimeValue", setVcatTimeValue);
        ARRUS_MATLAB_ADD_METHOD("setDtgcAttenuation", setDtgcAttenuation);
        ARRUS_MATLAB_ADD_METHOD("setActiveTermination", setActiveTermination);
        ARRUS_MATLAB_ADD_METHOD("setLnaGain", setLnaGain);
        ARRUS_MATLAB_ADD_METHOD("getLnaGain", getLnaGain);
        ARRUS_MATLAB_ADD_METHOD("setPgaGain", setPgaGain);
        ARRUS_MATLAB_ADD_METHOD("getPgaGain", getPgaGain);
        ARRUS_MATLAB_ADD_METHOD("setLpfCutoff", setLpfCutoff);
        ARRUS_MATLAB_ADD_METHOD("disableLnaHpf", disableLnaHpf);
        ARRUS_MATLAB_ADD_METHOD("setLnaHpfCornerFrequency", setLnaHpfCornerFrequency);
        ARRUS_MATLAB_ADD_METHOD("disableAdcHpf", disableAdcHpf);
        ARRUS_MATLAB_ADD_METHOD("setAdcHpfCornerFrequency", setAdcHpfCornerFrequency);

    }

    void disableHV(MatlabObjectHandle obj, MatlabOutputArgs &outputs, MatlabInputArgs &inputs) {
        ctx->logInfo("Us4R: disabling HV");
        get(obj)->disableHV();
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
        auto channelsMask = get(obj)->getChannelsMask(0);
        outputs[0] = ARRUS_MATLAB_GET_MATLAB_VECTOR(ctx, unsigned short, channelsMask);
    }

    void setTgcCurveValue(MatlabObjectHandle obj, MatlabOutputArgs &outputs, MatlabInputArgs &inputs) {
        std::vector<float> value = convertToCppVector<float>(inputs[0], "tgc values");
        bool applyCharacteristic = inputs[1][0];
        get(obj)->setTgcCurve(value, applyCharacteristic);
    }

    void setTgcCurveTimeValue(MatlabObjectHandle obj, MatlabOutputArgs &outputs, MatlabInputArgs &inputs) {
        std::vector<float> time = convertToCppVector<float>(inputs[0], "tgc time");
        std::vector<float> value = convertToCppVector<float>(inputs[1], "tgc values");
        bool applyCharacteristic = inputs[2][0];
        get(obj)->setTgcCurve(time, value, applyCharacteristic);
    }

    void setVcatTimeValue(MatlabObjectHandle obj, MatlabOutputArgs &outputs, MatlabInputArgs &inputs) {
        std::vector<float> time = convertToCppVector<float>(inputs[0], "tgc time");
        std::vector<float> value = convertToCppVector<float>(inputs[1], "attenuation values");
        bool applyCharacteristic = inputs[2][0];
        get(obj)->setVcat(time, value, applyCharacteristic);
    }

    void setDtgcAttenuation(MatlabObjectHandle obj, MatlabOutputArgs &outputs, MatlabInputArgs &inputs) {
        uint16 attenuation = inputs[0][0];
        get(obj)->setDtgcAttenuation(attenuation);
    }

    void setActiveTermination(MatlabObjectHandle obj, MatlabOutputArgs &outputs, MatlabInputArgs &inputs) {
        uint16 impedance = inputs[0][0];
        get(obj)->setActiveTermination(impedance);
    }

    void setLnaGain(MatlabObjectHandle obj, MatlabOutputArgs &outputs, MatlabInputArgs &inputs) {
        uint16 gain = inputs[0][0];
        get(obj)->setLnaGain(gain);
    }

    void getLnaGain(MatlabObjectHandle obj, MatlabOutputArgs &outputs, MatlabInputArgs &inputs) {
        float gain = get(obj)->getLnaGain();
        outputs[0] = ARRUS_MATLAB_GET_MATLAB_SCALAR(ctx, uint16, gain);
    }

    void setPgaGain(MatlabObjectHandle obj, MatlabOutputArgs &outputs, MatlabInputArgs &inputs) {
        uint16 gain = inputs[0][0];
        get(obj)->setPgaGain(gain);
    }

    void getPgaGain(MatlabObjectHandle obj, MatlabOutputArgs &outputs, MatlabInputArgs &inputs) {
        float gain = get(obj)->getPgaGain();
        outputs[0] = ARRUS_MATLAB_GET_MATLAB_SCALAR(ctx, uint16, gain);
    }

    void setLpfCutoff(MatlabObjectHandle obj, MatlabOutputArgs &outputs, MatlabInputArgs &inputs) {
        uint32 frequency = inputs[0][0];
        get(obj)->setLpfCutoff(frequency);
    }

    void disableLnaHpf(MatlabObjectHandle obj, MatlabOutputArgs &outputs, MatlabInputArgs &inputs) {
        ctx->logInfo("Us4R: disabling LNA HPF");
        get(obj)->disableLnaHpf();
    }

    void setLnaHpfCornerFrequency(MatlabObjectHandle obj, MatlabOutputArgs &outputs, MatlabInputArgs &inputs) {
        uint32 frequency = inputs[0][0];
        get(obj)->setLnaHpfCornerFrequency(frequency);
    }

    void disableAdcHpf(MatlabObjectHandle obj, MatlabOutputArgs &outputs, MatlabInputArgs &inputs) {
        ctx->logInfo("Us4R: disabling ADC HPF");
        get(obj)->disableAdcHpf();
    }

    void setAdcHpfCornerFrequency(MatlabObjectHandle obj, MatlabOutputArgs &outputs, MatlabInputArgs &inputs) {
        uint32 frequency = inputs[0][0];
        get(obj)->setAdcHpfCornerFrequency(frequency);
    }

};

}

#endif//API_MATLAB_WRAPPERS_DEVICES_US4R_US4RCLASSIMPL_H
