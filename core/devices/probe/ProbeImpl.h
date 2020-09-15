#ifndef ARRUS_CORE_DEVICES_PROBE_PROBEIMPL_H
#define ARRUS_CORE_DEVICES_PROBE_PROBEIMPL_H

#include "arrus/core/api/devices/probe/ProbeModel.h"
#include "arrus/core/api/devices/probe/Probe.h"
#include "arrus/core/api/devices/us4r/ProbeAdapter.h"
#include "arrus/core/devices/TxRxParameters.h"
#include "arrus/core/api/ops/us4r/tgc.h"
#include "arrus/core/common/logging.h"

namespace arrus::devices {

class ProbeImpl : public Probe {
public:
    ProbeImpl(const DeviceId &id, ProbeModel model,
              ProbeAdapter::RawHandle adapter,
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
    ProbeAdapter::RawHandle adapter;
    std::vector<ChannelIdx> channelMapping;
};

}

#endif //ARRUS_CORE_DEVICES_PROBE_PROBEIMPL_H
