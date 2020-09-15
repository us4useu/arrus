#include "ProbeImpl.h"

#include "arrus/common/asserts.h"

namespace arrus::devices {

ProbeImpl::ProbeImpl(const DeviceId &id, ProbeModel model,
                     ProbeAdapterImpl::RawHandle adapter,
                     std::vector<ChannelIdx> channelMapping)
    : Probe(id), logger{getLoggerFactory()->getLogger()},
      model(std::move(model)), adapter(adapter),
      channelMapping(std::move(channelMapping)) {

    INIT_ARRUS_DEVICE_LOGGER(logger, id.toString());
}


void ProbeImpl::setTxRxSequence(const std::vector<TxRxParameters> &seq,
                                const ops::us4r::TGCCurve &tgcSamples) {
    // TODO validate:
    // tx frequency should not exceed the maximum values
    // the size of rx and tx apertures should not exceed the (flattened) number of elements

    // set tx rx sequence
    std::vector<TxRxParameters> adapterSeq;

    auto probeNumberOfElements = model.getNumberOfElements().product();

    for(const auto &op: seq) {
        logger->log(LogSeverity::TRACE, arrus::format("Setting tx/rx {}", op));

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
                                op.getRxDecimationFactor(), op.getPri());
    }

    adapter->setTxRxSequence(adapterSeq, tgcSamples);

    // TODO call
}
}