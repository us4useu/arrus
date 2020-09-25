#include "Us4RImpl.h"

namespace arrus::devices {

void Us4RImpl::setTxRxSequence(const std::vector<TxRxParameters> &seq,
                               const ::arrus::ops::us4r::TGCCurve &tgcSamples) {
    getDefaultComponent()->setTxRxSequence(seq, tgcSamples);
}

void Us4RImpl::setVoltage(Voltage voltage) {
    ARRUS_REQUIRES_TRUE(hv.has_value(), "No HV have been set.");
    // Validate.
    auto *device = getDefaultComponent();
    auto voltageRange = device->getAcceptedVoltageRange();

    auto minVoltage = voltageRange.start();
    auto maxVoltage = voltageRange.end();

    if(voltage < minVoltage || voltage > maxVoltage) {
        throw IllegalArgumentException(
            ::arrus::format("Unaccepted voltage '{}', "
                            "should be in range: [{}, {}]",
                            voltage, minVoltage, maxVoltage));
    }
    hv.value()->setVoltage(voltage);
}

UltrasoundDevice *Us4RImpl::getDefaultComponent() {
    // NOTE! The implementation of this function determines
    // validation behaviour of SetVoltage function.
    // The safest option is to prefer using Probe only,
    // with an option to choose us4oem
    // (but the user has to specify it explicitly in settings).
    // Currently there should be no option to set TxRxSequence
    // on an adapter directly.
    if(probe.has_value()) {
        return probe.value().get();
    } else {
        return us4oems[0].get();
    }

}

void Us4RImpl::disableHV() {
    ARRUS_REQUIRES_TRUE(hv.has_value(), "No HV have been set.");
    hv.value()->disable();
}

}