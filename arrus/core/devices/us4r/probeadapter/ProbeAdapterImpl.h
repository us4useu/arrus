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
#include "arrus/core/devices/us4r/probeadapter/Us4OEMDataTransferRegistrar.h"
#include "arrus/core/api/ops/us4r/DigitalDownConversion.h"

namespace arrus::devices {

class ProbeAdapterImpl : public ProbeAdapterImplBase {
public:
    using Handle = std::unique_ptr<ProbeAdapterImpl>;
    using RawHandle = PtrHandle<ProbeAdapterImpl>;

    using ChannelAddress = ProbeAdapterSettings::ChannelAddress;
    using ChannelMapping = ProbeAdapterSettings::ChannelMapping;

    ProbeAdapterImpl(DeviceId deviceId, ProbeAdapterModelId modelId,std::vector<Us4OEMImplBase::RawHandle> us4oems,
                     ChannelIdx numberOfChannels, ChannelMapping channelMapping);

    [[nodiscard]] ChannelIdx getNumberOfChannels() const override {
        return numberOfChannels;
    }

    std::tuple<Us4RBuffer::Handle, EchoDataDescription::Handle>
    setTxRxSequence(const std::vector<TxRxParameters> &seq,
                    const ops::us4r::TGCCurve &tgcSamples, uint16 rxBufferSize=2,
                    uint16 rxBatchSize=1, std::optional<float> sri=std::nullopt,
                    bool triggerSync = false,
                    const std::optional<::arrus::ops::us4r::DigitalDownConversion> &ddc = std::nullopt) override;

    Ordinal getNumberOfUs4OEMs() override;

    void start() override;

    void stop() override;

    void syncTrigger() override;

    void registerOutputBuffer(Us4ROutputBuffer *buffer, const Us4RBuffer::Handle &us4rBuffer,
                              ::arrus::ops::us4r::Scheme::WorkMode workMode) override;

    void unregisterOutputBuffer() override;

private:
    void registerOutputBuffer(Us4ROutputBuffer *bufferDst, const Us4OEMBuffer &bufferSrc,
                              Us4OEMImplBase::RawHandle us4oem, ::arrus::ops::us4r::Scheme::WorkMode workMode);

    Us4OEMImplBase::RawHandle getMasterUs4oem() const {
        return this->us4oems[0];
    }
    size_t getUniqueUs4OEMBufferElementSize(const Us4OEMBuffer &us4oemBuffer) const;

    Logger::Handle logger;
    ProbeAdapterModelId modelId;
    std::vector<Us4OEMImplBase::RawHandle> us4oems;
    ChannelIdx numberOfChannels;
    ChannelMapping channelMapping;
    std::vector<std::shared_ptr<Us4OEMDataTransferRegistrar>> transferRegistrar;
};
}

#endif //ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERIMPL_H
