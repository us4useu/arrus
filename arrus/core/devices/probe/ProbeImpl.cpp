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

std::tuple<Us4RBuffer::Handle, EchoDataDescription::Handle>
ProbeImpl::setTxRxSequence(const std::vector<TxRxParameters> &seq, const ops::us4r::TGCCurve &tgcSamples,
                           uint16 rxBufferSize, uint16 rxBatchSize, std::optional<float> sri, bool triggerSync,
                           const std::optional<ops::us4r::DigitalDownConversion> &ddc) {
    // Validate input sequence
    ProbeTxRxValidator validator(format("tx rx sequence for {}", getDeviceId().toString()), model);
    validator.validate(seq);
    validator.throwOnErrors();

    // set tx rx sequence
    std::vector<TxRxParameters> adapterSeq;

    auto probeNumberOfElements = model.getNumberOfElements().product();

    // Each vector contains mapping:
    // probe's rx aperture element number -> adapter rx aperture channel number
    // Where each element is the active bit element/channel number.
    std::vector<std::vector<ChannelIdx>> rxApertureChannelMappings;

    // TODO the below list is used only in the remapFcm function, consider simplifying it
    std::vector<ChannelIdx> rxPaddingLeft;
    std::vector<ChannelIdx> rxPaddingRight;

    for (const auto &op: seq) {
        logger->log(LogSeverity::TRACE, format("Setting tx/rx {}", ::arrus::toString(op)));
        std::vector<ChannelIdx> rxApertureChannelMapping;

        BitMask txAperture(adapter->getNumberOfChannels());
        BitMask rxAperture(adapter->getNumberOfChannels());
        std::vector<float> txDelays(adapter->getNumberOfChannels());

        ARRUS_REQUIRES_TRUE(
            op.getTxAperture().size() == op.getRxAperture().size()
         && op.getTxAperture().size() == op.getTxDelays().size()
         && op.getTxAperture().size() == probeNumberOfElements,
            format("Probe's tx, rx apertures and tx delays array should have the same size: {}",
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
        adapterSeq.emplace_back(txAperture, txDelays, op.getTxPulse(), rxAperture, op.getRxSampleRange(),
                                op.getRxDecimationFactor(), op.getPri(), op.getRxPadding());
        rxApertureChannelMappings.push_back(rxApertureChannelMapping);

        rxPaddingLeft.push_back(op.getRxPadding()[0]);
        rxPaddingRight.push_back(op.getRxPadding()[1]);
    }

    auto[buffer, edd] = adapter->setTxRxSequence(adapterSeq, tgcSamples, rxBufferSize, rxBatchSize, sri, triggerSync,
                                                 ddc);
    FrameChannelMapping::Handle actualFcm = remapFcm(edd->fcm, rxApertureChannelMappings, rxPaddingLeft, rxPaddingRight);
    auto outEdd = std::make_shared<EchoDataDescription>(std::move(actualFcm), edd->rxOffset);
    return std::make_tuple(std::move(buffer), std::move(edd));
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

void ProbeImpl::registerOutputBuffer(Us4ROutputBuffer *buffer, const Us4RBuffer::Handle &us4rBuffer,
                                     ::arrus::ops::us4r::Scheme::WorkMode workMode) {
    adapter->registerOutputBuffer(buffer, us4rBuffer, workMode);
}

void ProbeImpl::unregisterOutputBuffer() {
    adapter->unregisterOutputBuffer();
}

// Remaps FCM according to given rx aperture active channels mappings.

/**
 * This function reorders channels in FCM produced by ProbeAdapterImpl, so the order of channel
 * is correct even in case of some permutation between probe and adapter channels (e.g. like
 * for ALS probes - esaote adapters).
 *
 * Basically, this function reads adapter FCM and sets the order channels according to the mapping
 * probe2AdpaterMap, which is
 * probe's aperture channel number -> adapter's aperture channel number
 *
 * e.g. in the case of probe-adapter mapping: 1-3, 2-1, 3-2, 3-element aperture,
 * the output FCM data (internal arrays) will be reordered 3, 1, 2 (i.e. probe's channel 1 will point to
 * adapter channel 3, and so on).
 */
FrameChannelMapping::Handle ProbeImpl::remapFcm(const FrameChannelMapping::Handle &adapterFcm,
                                                const std::vector<std::vector<ChannelIdx>> &adapterActiveChannels,
                                                const std::vector<ChannelIdx> &rxPaddingLeft,
                                                const std::vector<ChannelIdx> &rxPaddingRight) {
    auto nOps = adapterActiveChannels.size();
    if (adapterFcm->getNumberOfLogicalFrames() != nOps) {
        throw std::runtime_error("Inconsistent mapping and op number of probe's Rx apertures");
    }
    FrameChannelMappingBuilder builder = FrameChannelMappingBuilder::like(*adapterFcm);

    unsigned short frameNumber = 0;
    for (const auto &mapping : adapterActiveChannels) {
        // mapping[i] = dst adapter channel number
        // (e.g. from 0 to 256 (number of channels the system have))
        // where i is the probe rx active element
        // EXAMPLE: mapping = {3, 1, 10}
        auto paddingLeft = rxPaddingLeft[frameNumber];
        auto paddingRight = rxPaddingRight[frameNumber];

        // pairs: probe's APERTURE channel, adapter channel
        std::vector<std::pair<ChannelIdx, ChannelIdx>> posChannel;
        auto nRxChannels = mapping.size();
        // probe2AdapterMap[i] = dst adapter aperture channel number (e.g. from 0 to 64 (aperture size)).
        std::vector<ChannelIdx> probe2AdapterMap(nRxChannels, 0);

        std::transform(std::begin(mapping), std::end(mapping), std::back_insert_iterator(posChannel),
                       [i = 0](ChannelIdx channel) mutable {
                         return std::make_pair(static_cast<ChannelIdx>(i++), channel);
                       });
        // EXAMPLE: posChannel = {{0, 3}, {1, 1}, {2, 10}}
        std::sort(std::begin(posChannel), std::end(posChannel),
                  [](const auto &a, const auto &b) { return a.second < b.second; });
        // Now the position in the vector `posChannel` is equal to the adapter APERTURE channel.
        // EXAMPLE: posChannel = {{1, 1}, {0, 3}, {2, 10}}
        ChannelIdx i = 0;

        // probe aperture channel -> adapter aperture channel
        // EXAMPLE: probe2AdapterMap = {1, 0, 2}
        for (const auto& posCh: posChannel) {
            probe2AdapterMap[std::get<0>(posCh)] = i++;
        }
        // probe aperture rx number -> adapter aperture rx number -> physical channel
        auto nChannels = adapterFcm->getNumberOfLogicalChannels();
        for (ChannelIdx pch = 0; pch < nChannels; ++pch) {
            if(pch >= paddingLeft && pch < (nChannels-paddingRight)) {
                auto address = adapterFcm->getLogical(frameNumber, probe2AdapterMap[pch-paddingLeft]+paddingLeft);
                auto us4oem = address.getUs4oem();
                auto physicalFrame = address.getFrame();
                auto physicalChannel = address.getChannel();
                builder.setChannelMapping(frameNumber, pch, us4oem, physicalFrame, physicalChannel);
            }

        }
        ++frameNumber;
    }
    return builder.build();
}

}
