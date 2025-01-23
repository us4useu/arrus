#ifndef ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERIMPLBASE_H
#define ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERIMPLBASE_H

#include "arrus/core/api/devices/us4r/FrameChannelMapping.h"
#include "arrus/core/api/devices/us4r/ProbeAdapter.h"
#include "arrus/core/api/ops/us4r/Scheme.h"
#include "arrus/core/api/ops/us4r/tgc.h"
#include "arrus/core/devices/TxRxParameters.h"
#include "arrus/core/devices/us4r/DataTransfer.h"
#include "arrus/core/devices/us4r/Us4RBuffer.h"
#include "arrus/core/devices/us4r/Us4ROutputBuffer.h"

namespace arrus::devices {

class ProbeAdapterImplBase : public ProbeAdapter {
public:
    using ProbeAdapter::ProbeAdapter;

    using Handle = std::unique_ptr<ProbeAdapterImplBase>;
    using RawHandle = PtrHandle<ProbeAdapterImplBase>;

    virtual std::tuple<Us4RBuffer::Handle, FrameChannelMapping::Handle>
    setTxRxSequence(const std::vector<TxRxParameters> &seq, const ops::us4r::TGCCurve &tgcSamples, uint16 rxBufferSize,
                    uint16 rxBatchSize, std::optional<float> sri, arrus::ops::us4r::Scheme::WorkMode workMode,
                    const std::optional<ops::us4r::DigitalDownConversion> &ddc,
                    const std::vector<arrus::framework::NdArray> &txDelays) = 0;

    virtual Ordinal getNumberOfUs4OEMs() = 0;

    virtual void start() = 0;

    virtual void stop() = 0;

    virtual void syncTrigger() = 0;

    virtual std::tuple<Us4RBuffer::Handle, FrameChannelMapping::Handle>
    setSubsequence(uint16_t start, uint16_t end, const std::optional<float> &sri) = 0;
};

}// namespace arrus::devices

#endif//ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERIMPLBASE_H
