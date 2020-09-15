#include "ProbeAdapterImpl.h"

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

void ProbeAdapterImpl::setTxRxSequence(const std::vector<TxRxParameters> &seq,
                                       const TGCCurve &tgcSamples) {

    // TODO validation: tx and rx aperture should have exactly nchannels
    // TODO number of active tx/rx channels: will be verified by us4oems
    // (should not exceed the number of available channels)
    // Tx/Rx aperture should have the equal number of elements
    // Tx delays should contain exactly Tx aperture number of elements

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

    Ordinal us4oemOrdinal = 0;
    for(auto us4oem : us4oems) {
        std::vector<TxRxParameters> us4oemSeq;

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
        us4oem->setTxRxSequence(us4oemSeq, tgcSamples);
        // TODO What if tx aperture and rx aperture are empty?
        // Should be set - the same number of operations should be put
        // However, the data be ommitted if rx aperture is empty
        ++us4oemOrdinal;
    }
}

}