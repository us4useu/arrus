#ifndef ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERIMPLBASE_H
#define ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERIMPLBASE_H

#include "arrus/core/api/devices/us4r/ProbeAdapter.h"
#include "arrus/core/api/devices/us4r/FrameChannelMapping.h"
#include "arrus/core/api/ops/us4r/tgc.h"
#include "arrus/core/devices/TxRxParameters.h"
#include "arrus/core/devices/us4r/DataTransfer.h"

namespace arrus::devices {

class ProbeAdapterImplBase : public ProbeAdapter {
public:
    using ProbeAdapter::ProbeAdapter;

    using Handle = std::unique_ptr<ProbeAdapterImplBase>;
    using RawHandle = PtrHandle<ProbeAdapterImplBase>;

    virtual
    std::tuple<
        FrameChannelMapping::Handle,
        std::vector<std::vector<DataTransfer>>
    >
    setTxRxSequence(
        const std::vector<TxRxParameters> &seq,
        const ::arrus::ops::us4r::TGCCurve &tgcSamples) = 0;

    virtual Ordinal getNumberOfUs4OEMs() = 0;

    virtual void start() = 0;
    virtual void stop() = 0;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERIMPLBASE_H
