#include "ProbeImpl.h"

#include "arrus/common/asserts.h"
#include "arrus/core/common/validation.h"

namespace arrus::devices {

ProbeImpl::ProbeImpl(const DeviceId &id, ProbeModel model,
                     ProbeAdapterImplBase::RawHandle adapter,
                     std::vector<ChannelIdx> channelMapping)
    : ProbeImplBase(id), logger{getLoggerFactory()->getLogger()},
      model(std::move(model)), adapter(adapter),
      channelMapping(std::move(channelMapping)) {

    INIT_ARRUS_DEVICE_LOGGER(logger, id.toString());
}

class ProbeTxRxValidator : public Validator<TxRxParamsSequence> {
public:
    ProbeTxRxValidator(const std::string &componentName, const ProbeModel &modelRef)
        : Validator(componentName), modelRef(modelRef) {}

    void validate(const TxRxParamsSequence &txRxs) override {

        auto numberOfChannels = modelRef.getNumberOfElements().product();
        auto &txFrequencyRange = modelRef.getTxFrequencyRange();

        for(size_t firing = 0; firing < txRxs.size(); ++firing) {
            const auto &op = txRxs[firing];
            auto firingStr = ::arrus::format(" (firing {})", firing);
            ARRUS_VALIDATOR_EXPECT_EQUAL_M(
                op.getTxAperture().size(), numberOfChannels, firingStr);
            ARRUS_VALIDATOR_EXPECT_EQUAL_M(
                op.getRxAperture().size(), numberOfChannels, firingStr);
            ARRUS_VALIDATOR_EXPECT_EQUAL_M(
                op.getTxDelays().size(), numberOfChannels, firingStr);
            ARRUS_VALIDATOR_EXPECT_IN_RANGE_M(
                op.getTxPulse().getCenterFrequency(),
                txFrequencyRange.start(), txFrequencyRange.end(),
                firingStr);
        }
    }

private:
    const ProbeModel &modelRef;
};

std::tuple<
    FrameChannelMapping::Handle,
    std::vector<std::vector<DataTransfer>>
>
ProbeImpl::setTxRxSequence(const std::vector<TxRxParameters> &seq,
                                                       const ops::us4r::TGCCurve &tgcSamples) {
    // Validate input sequence
    ProbeTxRxValidator validator(
        ::arrus::format("tx rx sequence for {}", getDeviceId().toString()), model);
    validator.validate(seq);
    validator.throwOnErrors();

    // set tx rx sequence
    std::vector<TxRxParameters> adapterSeq;

    auto probeNumberOfElements = model.getNumberOfElements().product();

    for(const auto &op: seq) {
        logger->log(LogSeverity::TRACE, arrus::format(
            "Setting tx/rx {}", ::arrus::toString(op)));

        BitMask txAperture(adapter->getNumberOfChannels());
        BitMask rxAperture(adapter->getNumberOfChannels());
        std::vector<float> txDelays(adapter->getNumberOfChannels());

        ARRUS_REQUIRES_TRUE(
            op.getTxAperture().size() == op.getRxAperture().size()
            && op.getTxAperture().size() == op.getTxDelays().size()
            && op.getTxAperture().size() == probeNumberOfElements,
            arrus::format("Probe's tx, rx apertures and tx delays "
                          "array should have the same size: {}",
                          model.getNumberOfElements().product()));

        for(size_t pch = 0; pch < op.getTxAperture().size(); ++pch) {
            auto ach = channelMapping[pch];
            txAperture[ach] = op.getTxAperture()[pch];
            rxAperture[ach] = op.getRxAperture()[pch];
            txDelays[ach] = op.getTxDelays()[pch];
        }
        adapterSeq.emplace_back(txAperture, txDelays, op.getTxPulse(),
                                rxAperture, op.getRxSampleRange(),
                                op.getRxDecimationFactor(), op.getPri(),
                                op.getRxPadding(), op.getCallback());
    }

    return adapter->setTxRxSequence(adapterSeq, tgcSamples);
}

Interval<Voltage> ProbeImpl::getAcceptedVoltageRange() {
    return model.getVoltageRange();
}

void ProbeImpl::start() {
    adapter->start();
}

void ProbeImpl::stop() {
    adapter->stop();
}

}