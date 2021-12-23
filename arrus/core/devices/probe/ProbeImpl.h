#ifndef ARRUS_CORE_DEVICES_PROBE_PROBEIMPL_H
#define ARRUS_CORE_DEVICES_PROBE_PROBEIMPL_H

#include "arrus/core/api/devices/probe/ProbeModel.h"
#include "arrus/core/devices/probe/ProbeImplBase.h"
#include "arrus/core/api/devices/us4r/ProbeAdapter.h"
#include "arrus/core/devices/TxRxParameters.h"
#include "arrus/core/devices/UltrasoundDevice.h"
#include "arrus/core/api/ops/us4r/tgc.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/devices/us4r/probeadapter/ProbeAdapterImplBase.h"

namespace arrus::devices {

class ProbeImpl : public ProbeImplBase {
public:
    using Handle = std::unique_ptr<ProbeImpl>;
    using RawHandle = PtrHandle<ProbeImpl>;

    ProbeImpl(const DeviceId &id, ProbeModel model, ProbeAdapterImplBase::RawHandle adapter,
              std::vector<ChannelIdx> channelMapping);

    const ProbeModel &getModel() const override {
        return model;
    }

    std::tuple<Us4RBuffer::Handle, FrameChannelMapping::Handle>
    setTxRxSequence(const std::vector<TxRxParameters> &seq, const ops::us4r::TGCCurve &tgcSamples, uint16 rxBufferSize,
                    uint16 rxBatchSize, std::optional<float> sri, bool triggerSync = false) override;

    Interval<Voltage> getAcceptedVoltageRange() override;

    void start() override;

    void stop() override;

    void syncTrigger() override;

    void registerOutputBuffer(Us4ROutputBuffer *buffer, const Us4RBuffer::Handle &us4rBuffer,
                              ::arrus::ops::us4r::Scheme::WorkMode workMode) override;

    void unregisterOutputBuffer(Us4ROutputBuffer *buffer, const Us4RBuffer::Handle &handle) override;

    static FrameChannelMapping::Handle remapFcm(const FrameChannelMapping::Handle &adapterFcm,
                                                const std::vector<std::vector<ChannelIdx>> &adapterActiveChannels,
                                                const std::vector<ChannelIdx> &rxPaddingLeft,
                                                const std::vector<ChannelIdx> &rxPaddingRight);

private:
    Logger::Handle logger;
    ProbeModel model;
    ProbeAdapterImplBase::RawHandle adapter;
    std::vector<ChannelIdx> channelMapping;

};

}

#endif //ARRUS_CORE_DEVICES_PROBE_PROBEIMPL_H
