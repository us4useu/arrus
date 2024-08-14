#ifndef ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERIMPL_H
#define ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERIMPL_H

#include <utility>

#include "arrus/common/asserts.h"
#include "arrus/core/api/devices/us4r/ProbeAdapterSettings.h"
#include "arrus/core/api/ops/us4r/DigitalDownConversion.h"
#include "arrus/core/api/ops/us4r/tgc.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/devices/TxRxParameters.h"
#include "arrus/core/devices/us4r/FrameChannelMappingImpl.h"
#include "arrus/core/devices/us4r/Us4OEMDataTransferRegistrar.h"
#include "arrus/core/devices/us4r/Us4RBuffer.h"
#include "arrus/core/devices/us4r/probeadapter/ProbeAdapterImplBase.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMImpl.h"

namespace arrus::devices {

class ProbeAdapterImpl : public ProbeAdapterImplBase {
public:
    using Handle = std::unique_ptr<ProbeAdapterImpl>;
    using RawHandle = PtrHandle<ProbeAdapterImpl>;

    using ChannelAddress = ProbeAdapterSettings::ChannelAddress;
    using ChannelMapping = ProbeAdapterSettings::ChannelMapping;

    ProbeAdapterImpl(DeviceId deviceId, ProbeAdapterModelId modelId,std::vector<Us4OEMImplBase::RawHandle> us4oems,
                     ChannelIdx numberOfChannels, ChannelMapping channelMapping,
                     const ::arrus::devices::us4r::IOSettings &ioSettings);

    [[nodiscard]] ChannelIdx getNumberOfChannels() const override {
        return numberOfChannels;
    }

    std::tuple<Us4RBuffer::Handle, FrameChannelMapping::Handle>
    setTxRxSequence(const std::vector<TxRxParameters> &seq,
                    const ops::us4r::TGCCurve &tgcSamples, uint16 rxBufferSize=2,
                    uint16 rxBatchSize=1, std::optional<float> sri=std::nullopt,
                    bool triggerSync = false,
                    const std::optional<ops::us4r::DigitalDownConversion> &ddc = std::nullopt,
                    const std::vector<framework::NdArray> &txDelays = std::vector<arrus::framework::NdArray>()
                    ) override;

    Ordinal getNumberOfUs4OEMs() override;

    void start() override;

    void stop() override;

    void syncTrigger() override;

    std::tuple<Us4RBuffer::Handle, FrameChannelMapping::Handle>
    setSubsequence(uint16_t start, uint16_t end, const std::optional<float> &sri) override;

private:
    struct OpToNextFrameMapping {
        OpToNextFrameMapping(uint16_t nFirings, const std::vector<Us4OEMBufferElementPart> &frames);

        std::optional<uint16> getNextFrame(uint16 op) {
            if(op >= opToNextFrame.size()) {
                throw IllegalArgumentException("Accessing mapping outside the avialable range.");
            }
            return opToNextFrame.at(op);
        }

        /**
         * Returns the number of frames acquired by ops with numbers between [start, end] (both inclusive).
         */
        long getNumberOfFrames(uint16 start, uint16 end) {
            if(start > end || end >= isRxOp.size()) {
                throw std::runtime_error("Accessing isRxOp outside the available range.");
            }
            long result = 0;
            for(uint16 i = start; i <= end; ++i) {
                if(isRxOp.at(i)) {
                    ++result;
                }
            }
            return result;
        }
        // op (firing) number -> next frame number, relative to the full sequence.
        std::vector<std::optional<uint16_t>> opToNextFrame;
        // op (firing) number -> whether there is some data acquistion done by this op
        std::vector<bool> isRxOp;
    };


    void calculateRxDelays(std::vector<TxRxParamsSequence> &sequences);
    Ordinal getFrameMetadataOem(const us4r::IOSettings &settings);

    Logger::Handle logger;
    ProbeAdapterModelId modelId;
    std::vector<Us4OEMImplBase::RawHandle> us4oems;
    ChannelIdx numberOfChannels;
    ChannelMapping channelMapping;
    /** The OEM, which is responsible for acquiring pulse counter metadata (ordinal number). **/
    Ordinal frameMetadataOem{0};

    // Subsequence selection properties.
    /** Logical -> physical [start, end] op (TX/RX) */
    std::vector<std::pair<uint16_t, uint16_t>> logicalToPhysicalOp;
    std::vector<Us4OEMBuffer> fullSequenceOEMBuffers;
    /** OEM number -> physical op -> next frame number (from the complete frame sequence) */
    std::vector<OpToNextFrameMapping> physicalOpToNextFrame;
    FrameChannelMappingImpl::Handle fullSequenceFCM;
    /** Sequencer start pointer that should be set in the next call of the start method. NOTE: this property
        will usually be set to 0, except the case where the setting Seqeuncer pointer to 0 is not acceptable
        e.g. after calling setSubsequence method with start > 0. */
    uint16 oemSequencerStartEntry{0};
    bool isCurrentlyTriggerSync{false};
};
}

#endif //ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERIMPL_H
