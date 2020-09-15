#ifndef ARRUS_CORE_DEVICES_PROBE_PROBEIMPL_H
#define ARRUS_CORE_DEVICES_PROBE_PROBEIMPL_H

#include "arrus/core/api/devices/probe/ProbeModel.h"
#include "arrus/core/api/devices/probe/Probe.h"
#include "arrus/core/api/devices/us4r/ProbeAdapter.h"
#include "arrus/core/devices/TxRxParameters.h"
#include "arrus/core/api/ops/us4r/tgc.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/devices/us4r/probeadapter/ProbeAdapterImpl.h"

namespace arrus::devices {

class ProbeImpl : public Probe {
public:
    using Handle = std::unique_ptr<ProbeImpl>;
    using RawHandle = PtrHandle<ProbeImpl>;

    ProbeImpl(const DeviceId &id, ProbeModel model,
              ProbeAdapterImpl::RawHandle adapter,
              std::vector<ChannelIdx> channelMapping);

    /**
     * tx and rx aperture are expected to be provided in flattened format.
     *
     * @param seq
     * @param tgcSamples
     */
    void setTxRxSequence(const std::vector<TxRxParameters> &seq,
                         const ::arrus::ops::us4r::TGCCurve &tgcSamples);

private:
    Logger::Handle logger;
    ProbeModel model;
    ProbeAdapterImpl::RawHandle adapter;
    std::vector<ChannelIdx> channelMapping;
};

}

#endif //ARRUS_CORE_DEVICES_PROBE_PROBEIMPL_H
