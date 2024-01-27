#ifndef ARRUS_CORE_DEVICES_US4R_MAPPING_INTERFACETOINTERFACEMAPPINGCONVERTER_H
#define ARRUS_CORE_DEVICES_US4R_MAPPING_INTERFACETOINTERFACEMAPPINGCONVERTER_H

#include <utility>

#include "arrus/core/api/devices/us4r/FrameChannelMapping.h"
#include "arrus/core/api/framework/NdArray.h"
#include "arrus/core/devices/us4r/validators/ProbeTxRxValidator.h"

namespace arrus::devices::us4r {

using namespace ::arrus::framework;
using namespace ::arrus::devices;

class ProbeToAdapterMappingConverter {
public:
    /**
     * Probe <-> adapter (and adapter to probe) mapping converter.
     */
    ProbeToAdapterMappingConverter(const DeviceId &probeTxId, const DeviceId &probeRxId, ProbeSettings probeTx,
                                   ProbeSettings probeRx, const ChannelIdx adapterNChannels)
        : probeTxId(probeTxId), probeRxId(probeRxId), probeTx(std::move(probeTx)), probeRx(std::move(probeRx)),
          adapterNChannels(adapterNChannels) {}

    std::pair<TxRxParametersSequence, std::vector<NdArray>>
    convert(SequenceId id, const TxRxParametersSequence &sequence, const std::vector<NdArray> &txDelayProfiles) {
        // Validate input sequence
        ProbeTxRxValidator validator(format("Probe to adapter conversion, sequence: {}", id), probeTx.getModel(),
                                     probeRx.getModel());
        validator.validate(sequence);
        validator.throwOnErrors();

        // set tx rx sequence
        OpId nOps = ARRUS_SAFE_CAST(sequence.size(), OpId);
        TxRxParametersSequenceBuilder seqBuilder;
        // std::vector<TxRxParameters> adapterSeq;
        std::vector<NdArray> adapterTxDelayProfiles;
        NdArray::Shape outputProfileShape = {nOps, adapterNChannels};
        for (auto &inputTxDelayProfile : txDelayProfiles) {
            NdArray emptyArray(outputProfileShape, inputTxDelayProfile.getDataType(),
                               inputTxDelayProfile.getPlacement(), inputTxDelayProfile.getName());
            adapterTxDelayProfiles.push_back(emptyArray);
        }
        auto nElementsTx = probeTx.getModel().getNumberOfElements().product();

        size_t opIdx = 0;
        // Probe sequence -> adapter sequence
        for (const auto &op : sequence.getParameters()) {
            TxRxParametersBuilder paramBuilder(op);
            std::vector<ChannelIdx> rxApertureChannelMapping;

            // Adapte channel apertures/delays.
            BitMask txAperture(adapterNChannels);
            BitMask rxAperture(adapterNChannels);
            std::vector<float> txDelays(adapterNChannels);

            ARRUS_REQUIRES_TRUE(
                op.getTxAperture().size() == op.getTxDelays().size() && op.getTxAperture().size() == nElementsTx,
                format("Probe's tx, rx apertures and tx delays array should have the same size: {}", nElementsTx));

            // TX
            for (size_t pch = 0; pch < op.getTxAperture().size(); ++pch) {
                auto achTx = probeTx.getChannelMapping().at(pch);
                txAperture[achTx] = op.getTxAperture()[pch];
                txDelays[achTx] = op.getTxDelays()[pch];
                size_t nTxDelayProfiles = txDelayProfiles.size();
                for (size_t i = 0; i < nTxDelayProfiles; ++i) {
                    adapterTxDelayProfiles[i].set(opIdx, achTx, txDelayProfiles[i].get<float>(opIdx, pch));
                }
            }
            // RX
            for (size_t pch = 0; pch < op.getTxAperture().size(); ++pch) {
                auto achRx = probeRx.getChannelMapping().at(pch);
                rxAperture[achRx] = op.getRxAperture()[pch];
                if (op.getRxAperture()[pch]) {
                    rxApertureChannelMapping.push_back(achRx);
                }
            }
            paramBuilder.setTxAperture(txAperture);
            paramBuilder.setTxDelays(txDelays);
            paramBuilder.setRxAperture(rxAperture);
            seqBuilder.addEntry(paramBuilder.build());

            if (!op.isRxNOP()) {
                adapterActiveRxChannels.push_back(rxApertureChannelMapping);
                rxPaddingLeft.push_back(op.getRxPadding()[0]);
                rxPaddingRight.push_back(op.getRxPadding()[1]);
            }
            ++opIdx;
        }
        return std::make_pair(std::move(seqBuilder.build()), std::move(adapterTxDelayProfiles));
    }

    FrameChannelMapping::Handle convert(const FrameChannelMapping::Handle &adapterFCM) {
        auto nOps = adapterActiveRxChannels.size();
        if (adapterFCM->getNumberOfLogicalFrames() != nOps) {
            throw std::runtime_error("Inconsistent mapping and op number of probe's Rx apertures");
        }
        FrameChannelMappingBuilder builder = FrameChannelMappingBuilder::like(*adapterFCM);

        unsigned short frameNumber = 0;
        for (const auto &mapping : adapterActiveRxChannels) {
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

            std::transform(
                std::begin(mapping), std::end(mapping), std::back_insert_iterator(posChannel),
                [i = 0](ChannelIdx channel) mutable { return std::make_pair(static_cast<ChannelIdx>(i++), channel); });
            // EXAMPLE: posChannel = {{0, 3}, {1, 1}, {2, 10}}
            std::sort(std::begin(posChannel), std::end(posChannel),
                      [](const auto &a, const auto &b) { return a.second < b.second; });
            // Now the position in the vector `posChannel` is equal to the adapter APERTURE channel.
            // EXAMPLE: posChannel = {{1, 1}, {0, 3}, {2, 10}}
            ChannelIdx i = 0;

            // probe aperture channel -> adapter aperture channel
            // EXAMPLE: probe2AdapterMap = {1, 0, 2}
            for (const auto &posCh : posChannel) {
                probe2AdapterMap[std::get<0>(posCh)] = i++;
            }
            // probe aperture rx number -> adapter aperture rx number -> physical channel
            auto nChannels = adapterFCM->getNumberOfLogicalChannels();
            for (ChannelIdx pch = 0; pch < nChannels; ++pch) {
                if (pch >= paddingLeft && pch < (nChannels - paddingRight)) {
                    auto address =
                        adapterFCM->getLogical(frameNumber, probe2AdapterMap[pch - paddingLeft] + paddingLeft);
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

private:
    DeviceId probeTxId, probeRxId;
    ProbeSettings probeTx, probeRx;
    ChannelIdx adapterNChannels;

    // FCM temporary objects.
    // Each vector contains mapping:
    // probe's rx aperture element number -> ADAPTER rx channel number
    // Where each element is the active bit element/channel number.
    // NOTE: the target value is the GLOBAL ADAPTER channel number, not the ADAPTER CHANNEL NUMBER
    std::vector<std::vector<ChannelIdx>> adapterActiveRxChannels;
    std::vector<ChannelIdx> rxPaddingLeft;
    std::vector<ChannelIdx> rxPaddingRight;
};
}// namespace arrus::devices::us4r

#endif// ARRUS_CORE_DEVICES_US4R_MAPPING_INTERFACETOINTERFACEMAPPINGCONVERTER_H