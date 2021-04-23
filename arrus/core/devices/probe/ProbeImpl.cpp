#include <utility>

#include "ProbeImpl.h"

#include "arrus/common/asserts.h"
#include "arrus/core/common/validation.h"
#include "arrus/core/devices/us4r/FrameChannelMappingImpl.h"

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
  ProbeTxRxValidator(const std::string &componentName,
                     const ProbeModel &modelRef)
      : Validator(componentName), modelRef(modelRef) {}

  void validate(const TxRxParamsSequence &txRxs) override {

      auto numberOfChannels = modelRef.getNumberOfElements().product();
      auto &txFrequencyRange = modelRef.getTxFrequencyRange();

      for (size_t firing = 0; firing < txRxs.size(); ++firing) {
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

std::tuple<Us4RBuffer::Handle, FrameChannelMapping::Handle>
ProbeImpl::setTxRxSequence(const std::vector<TxRxParameters> &seq,
                           const ops::us4r::TGCCurve &tgcSamples,
                           uint16 rxBufferSize,
                           uint16 rxBatchSize, std::optional<float> sri,
                           bool triggerSync) {
    // Validate input sequence
    ProbeTxRxValidator validator(
        ::arrus::format("tx rx sequence for {}", getDeviceId().toString()),
        model);
    validator.validate(seq);
    validator.throwOnErrors();

    // set tx rx sequence
    std::vector<TxRxParameters> adapterSeq;

    auto probeNumberOfElements = model.getNumberOfElements().product();

    // Each vector contains mapping:
    // probe's rx aperture element number -> adapter rx aperture channel number
    // Where each element and channel is the active bit element/channel number.
    std::vector<std::vector<ChannelIdx>> rxApertureChannelMappings;
    for (const auto &op: seq) {
        logger->log(LogSeverity::TRACE, arrus::format(
            "Setting tx/rx {}", ::arrus::toString(op)));
        std::vector<ChannelIdx> rxApertureChannelMapping;

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

        for (size_t pch = 0; pch < op.getTxAperture().size(); ++pch) {
            auto ach = channelMapping[pch];
            txAperture[ach] = op.getTxAperture()[pch];
            rxAperture[ach] = op.getRxAperture()[pch];
            txDelays[ach] = op.getTxDelays()[pch];

            if (op.getRxAperture()[pch]) {
                rxApertureChannelMapping.push_back(ach);
            }
        }
        adapterSeq.emplace_back(txAperture, txDelays, op.getTxPulse(),
                                rxAperture, op.getRxSampleRange(),
                                op.getRxDecimationFactor(), op.getPri(),
                                op.getRxPadding());
        rxApertureChannelMappings.push_back(rxApertureChannelMapping);
    }

    auto[buffer, fcm] = adapter->setTxRxSequence(adapterSeq, tgcSamples,
                                                 rxBufferSize, rxBatchSize,
                                                 sri, triggerSync);
    FrameChannelMapping::Handle actualFcm = remapFcm(
        fcm, rxApertureChannelMappings);
    return std::make_tuple(std::move(buffer), std::move(actualFcm));
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

void ProbeImpl::syncTrigger() {
    adapter->syncTrigger();
}

void ProbeImpl::registerOutputBuffer(Us4ROutputBuffer *buffer,
                                     const Us4RBuffer::Handle &us4rBuffer,
                                     bool isTriggerSync) {
    adapter->registerOutputBuffer(buffer, us4rBuffer, isTriggerSync);
}

void ProbeImpl::setTgcCurve(const std::vector<float> &tgcCurve) {
    adapter->setTgcCurve(tgcCurve);
}

// Remaps FCM according to given rx aperture active channels mappings.
FrameChannelMapping::Handle ProbeImpl::remapFcm(
    const FrameChannelMapping::Handle &adapterFcm,
    const std::vector<std::vector<ChannelIdx>> &adapterActiveChannels)
{
    auto nOps = adapterActiveChannels.size();
    if (adapterFcm->getNumberOfLogicalFrames() != nOps) {
        throw std::runtime_error(
            "Inconsistent mapping and op number of probe's Rx apertures");
    }
    FrameChannelMappingBuilder builder(adapterFcm->getNumberOfLogicalFrames(),
                                       adapterFcm->getNumberOfLogicalChannels());

    unsigned short frameNumber = 0;
    for (const auto &mapping : adapterActiveChannels) {
        // mapping[i] = dst probe adapter channel number
        // (e.g. from 0 to 256 (number of channels system have))
        // where i is the probe rx active element

        // pairs: channel position, adapter channel
        std::vector<std::pair<ChannelIdx, ChannelIdx>> posChannel;
        size_t rxChannels = mapping.size();
        // adapterRxChannel[i] = dst adapter aperture channel number
        // (e.g. from 0 to 64 (aperture size)).
        std::vector<ChannelIdx> adapterRxChannel(rxChannels, 0);

        std::transform(std::begin(mapping), std::end(mapping),
                       std::back_insert_iterator(posChannel),
                       [i = 0](ChannelIdx channel) mutable {
                         return std::make_pair(static_cast<ChannelIdx>(i++), channel);
                       });
        std::sort(std::begin(posChannel), std::end(posChannel),
                  [](const auto &a, const auto &b) { return a.second < b.second; });
        ChannelIdx i = 0;
        for (const auto& pos_ch: posChannel) {
            adapterRxChannel[std::get<0>(pos_ch)] = i++;
        }
        // probe aperture rx number -> adapter aperture rx number ->
        // physical channel
        for (ChannelIdx pch = 0; pch < rxChannels; ++pch) {
            auto[physicalFrame, physicalChannel] =
            adapterFcm->getLogical(frameNumber, adapterRxChannel[pch]);
            builder.setChannelMapping(frameNumber, pch,
                                      physicalFrame, physicalChannel);
        }
        ++frameNumber;
    }
    return builder.build();
}

}
