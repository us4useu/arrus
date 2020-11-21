#ifndef ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERIMPLBASE_H
#define ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERIMPLBASE_H

#include "arrus/core/api/devices/us4r/ProbeAdapter.h"
#include "arrus/core/api/devices/us4r/FrameChannelMapping.h"
#include "arrus/core/api/ops/us4r/tgc.h"
#include "arrus/core/devices/TxRxParameters.h"
#include "arrus/core/devices/us4r/DataTransfer.h"
#include "arrus/core/api/devices/us4r/HostBuffer.h"
#include "arrus/core/devices/us4r/Us4ROutputBuffer.h"

namespace arrus::devices {

class ProbeAdapterImplBase : public ProbeAdapter {
public:
    using ProbeAdapter::ProbeAdapter;

    using Handle = std::unique_ptr<ProbeAdapterImplBase>;
    using RawHandle = PtrHandle<ProbeAdapterImplBase>;

    virtual
    std::tuple<
        FrameChannelMapping::Handle,
        std::vector<std::vector<DataTransfer>>,
        float // total PRI
    >
    setTxRxSequence(const std::vector<TxRxParameters> &seq,
                    const ::arrus::ops::us4r::TGCCurve &tgcSamples,
                    uint16 rxBufferSize, uint16 rxBatchSize,
                    std::optional<float> frameRepetitionInterval) = 0;

    virtual Ordinal getNumberOfUs4OEMs() = 0;

    virtual void start() = 0;

    virtual void stop() = 0;

    virtual void syncTrigger() = 0;

    virtual void registerOutputBuffer(Us4ROutputBuffer* buffer,
                                    const std::vector<std::vector<DataTransfer>> &transfers) = 0;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERIMPLBASE_H
