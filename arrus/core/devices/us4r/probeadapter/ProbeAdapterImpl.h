#ifndef ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERIMPL_H
#define ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERIMPL_H

#include <utility>

#include "arrus/core/devices/us4r/probeadapter/ProbeAdapterImplBase.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMImpl.h"
#include "arrus/core/api/devices/us4r/ProbeAdapterSettings.h"
#include "arrus/core/devices/TxRxParameters.h"
#include "arrus/core/api/ops/us4r/tgc.h"
#include "arrus/common/asserts.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/devices/us4r/Us4RBuffer.h"

namespace arrus::devices {

class ProbeAdapterImpl : public ProbeAdapterImplBase {
public:
    using Handle = std::unique_ptr<ProbeAdapterImpl>;
    using RawHandle = PtrHandle<ProbeAdapterImpl>;

    using ChannelAddress = ProbeAdapterSettings::ChannelAddress;
    using ChannelMapping = ProbeAdapterSettings::ChannelMapping;

    ProbeAdapterImpl(DeviceId deviceId, ProbeAdapterModelId modelId,
                     std::vector<Us4OEMImplBase::RawHandle> us4oems,
                     ChannelIdx numberOfChannels,
                     ChannelMapping channelMapping);

    [[nodiscard]] ChannelIdx getNumberOfChannels() const override {
        return numberOfChannels;
    }

    std::tuple<Us4RBuffer::Handle, FrameChannelMapping::Handle>
    setTxRxSequence(const std::vector<TxRxParameters> &seq,
                    const ::arrus::ops::us4r::TGCCurve &tgcSamples,
                    uint16 rxBufferSize = 1, uint16 rxBatchSize = 1,
                    std::optional<float> sri = std::nullopt) override;

    Ordinal getNumberOfUs4OEMs() override;

    void start() override;

    void stop() override;

    void syncTrigger() override;

    void registerOutputBuffer(Us4ROutputBuffer *buffer, const Us4RBuffer::Handle &transfers);

    void setTgcCurve(const arrus::ops::us4r::TGCCurve &curve) override;

private:
    Logger::Handle logger;
    ProbeAdapterModelId modelId;
    std::vector<Us4OEMImplBase::RawHandle> us4oems;
    ChannelIdx numberOfChannels;
    ChannelMapping channelMapping;

    void registerOutputBuffer(Us4ROutputBuffer *outputBuffer, const Us4OEMBuffer &us4oemBuffer, Us4OEMImplBase::RawHandle us4oem);

    Us4OEMImplBase::RawHandle getMasterUs4oem() const {
        return this->us4oems[0];
    }
};

}

#endif //ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERIMPL_H
