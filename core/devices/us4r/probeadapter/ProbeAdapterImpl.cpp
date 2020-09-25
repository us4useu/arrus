#include "ProbeAdapterImpl.h"

#include "arrus/core/devices/us4r/common.h"
#include "arrus/core/common/validation.h"

namespace arrus::devices {

using namespace ::arrus::ops::us4r;

ProbeAdapterImpl::ProbeAdapterImpl(DeviceId deviceId,
                                   ProbeAdapterModelId modelId,
                                   std::vector<Us4OEMImpl::RawHandle> us4oems,
                                   ChannelIdx numberOfChannels,
                                   ChannelMapping channelMapping)
    : ProbeAdapter(deviceId), logger(getLoggerFactory()->getLogger()),
      modelId(std::move(modelId)),
      us4oems(std::move(us4oems)),
      numberOfChannels(numberOfChannels),
      channelMapping(std::move(channelMapping)) {

    INIT_ARRUS_DEVICE_LOGGER(logger, id.toString());
}

class ProbeAdapterTxRxValidator : public Validator<TxRxParamsSequence> {
public:
    ProbeAdapterTxRxValidator(const std::string &componentName, ChannelIdx nChannels)
        : Validator(componentName), nChannels(nChannels) {}

    void validate(const TxRxParamsSequence &txRxs) override {
        for(size_t firing = 0; firing < txRxs.size(); ++firing) {
            const auto &op = txRxs[firing];
            auto firingStr = ::arrus::format("firing {}", firing);
            ARRUS_VALIDATOR_EXPECT_EQUAL_M(
                op.getRxAperture().size(), size_t(nChannels), firingStr);
            ARRUS_VALIDATOR_EXPECT_EQUAL_M(
                op.getTxAperture().size(), size_t(nChannels), firingStr);
            ARRUS_VALIDATOR_EXPECT_EQUAL_M(
                op.getTxDelays().size(), size_t(nChannels), firingStr);
        }
    }

private:
    ChannelIdx nChannels;
};


void ProbeAdapterImpl::setTxRxSequence(const std::vector<TxRxParameters> &seq,
                                       const TGCCurve &tgcSamples) {
    // Validate input sequence
    ProbeAdapterTxRxValidator validator(
        ::arrus::format("{} tx rx sequence", getDeviceId().toString()), numberOfChannels);
    validator.validate(seq);
    validator.throwOnErrors();

    // Split into multiple arrays.
    // us4oem, op number -> aperture/delays
    std::unordered_map<Ordinal, std::vector<BitMask>> txApertures, rxApertures;
    std::unordered_map<Ordinal, std::vector<std::vector<float>>> txDelaysList;

    // Initialize helper arrays.
    for(size_t ordinal = 0; ordinal < us4oems.size(); ++ordinal) {
        txApertures.emplace(ordinal, std::vector<BitMask>(seq.size()));
        rxApertures.emplace(ordinal, std::vector<BitMask>(seq.size()));
        txDelaysList.emplace(ordinal,
                             std::vector<std::vector<float>>(seq.size()));
    }

    // Split Tx, Rx apertures and tx delays into sub-apertures specific for
    // each us4oem module.
    uint32 opNumber = 0;
    for(const auto &op : seq) {
        logger->log(LogSeverity::TRACE, arrus::format("Setting tx/rx {}", op));

        const auto &txAperture = op.getTxAperture();
        const auto &rxAperture = op.getRxAperture();
        const auto &txDelays = op.getTxDelays();
        ARRUS_REQUIRES_TRUE(txAperture.size() == rxAperture.size()
                            && txAperture.size() == numberOfChannels,
                            arrus::format(
                                "Tx and Rx apertures should have a size: {}",
                                numberOfChannels));

        for(size_t ordinal = 0; ordinal < us4oems.size(); ++ordinal) {
            txApertures[ordinal][opNumber].resize(Us4OEMImpl::N_ADDR_CHANNELS);
            rxApertures[ordinal][opNumber].resize(Us4OEMImpl::N_ADDR_CHANNELS);
            txDelaysList[ordinal][opNumber].resize(Us4OEMImpl::N_ADDR_CHANNELS);
        }

        for(size_t ach = 0; ach < numberOfChannels; ++ach) {
            const auto[dstModule, dstChannel] = channelMapping[ach];
            txApertures[dstModule][opNumber][dstChannel] = op.getTxAperture()[ach];
            rxApertures[dstModule][opNumber][dstChannel] = op.getRxAperture()[ach];
            txDelaysList[dstModule][opNumber][dstChannel] = op.getTxDelays()[ach];
        }
        ++opNumber;
    }


    // Create operations for each of the us4oem module.
    std::vector<TxRxParamsSequence> seqs(us4oems.size());

    Ordinal us4oemOrdinal = 0;
    for(auto &us4oem : us4oems) {
        auto &us4oemSeq = seqs[us4oemOrdinal];

        uint16 i = 0;
        for(const auto &op : seq) {
            constexpr ChannelIdx N_CHANNELS = Us4OEMImpl::N_ADDR_CHANNELS;
            // Convert tx aperture to us4oem tx aperture

            const auto &txAperture = txApertures[us4oemOrdinal][i];
            const auto &rxAperture = rxApertures[us4oemOrdinal][i];
            const auto &txDelays = txDelaysList[us4oemOrdinal][i];

            us4oemSeq.emplace_back(txAperture, txDelays, op.getTxPulse(),
                                   rxAperture, op.getRxSampleRange(),
                                   op.getRxDecimationFactor(), op.getPri());
            ++i;
        }
        // What if tx aperture and rx aperture are empty?
        // Should be set - the same number of operations should be put for each module.
        // However, the data should be omitted if rx aperture is empty
        ++us4oemOrdinal;
    }
    // split operations if necessary
    std::vector<TxRxParamsSequence> sanitizedSeqs = splitRxAperturesIfNecessary(seqs);

    // set sequence on each us4oem
    us4oemOrdinal = 0;
    for(auto &us4oem : us4oems) {
        us4oem->setTxRxSequence(sanitizedSeqs[us4oemOrdinal], tgcSamples);
        ++us4oemOrdinal;
    }
}

}