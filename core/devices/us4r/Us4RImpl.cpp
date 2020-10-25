#include "Us4RImpl.h"

namespace arrus::devices {

using ::arrus::ops::us4r::TxRxSequence;
using ::arrus::ops::us4r::Tx;
using ::arrus::ops::us4r::Rx;
using ::arrus::ops::us4r::Pulse;


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

void Us4RImpl::upload(const ops::us4r::TxRxSequence &seq) {


    // translate the sequence to a double TxRxSequence
    // The first part of the sequence gathers data to the first buffer
    // the second one to the second buffer.
    std::vector<TxRxParameters> actualSeq;

    for(const auto& [tx, rx] : seq.getOps()) {
        actualSeq.emplace_back(
            tx.getAperture(),
            tx.getDelays(),
            tx.getExcitation(),
            rx.getAperture(),
            rx.getSampleRange(),
            rx.getDownsamplingFactor(),
            seq.getPri(),
            rx.getPadding()
        );
    }
    getDefaultComponent()->setTxRxSequence(actualSeq);
    // TODO return the created buffer
}

void Us4RImpl::start() {

}

void Us4RImpl::stop() {

}

}