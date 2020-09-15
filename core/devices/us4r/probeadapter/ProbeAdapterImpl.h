#ifndef ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERIMPL_H
#define ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERIMPL_H

#include <utility>

#include "arrus/core/api/devices/us4r/ProbeAdapter.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMImpl.h"
#include "arrus/core/api/devices/us4r/ProbeAdapterSettings.h"
#include "arrus/core/devices/TxRxParameters.h"
#include "arrus/core/api/ops/us4r/tgc.h"
#include "arrus/common/asserts.h"
#include "arrus/core/common/logging.h"


namespace arrus::devices {

class ProbeAdapterImpl : public ProbeAdapter {
public:
    using ChannelAddress = ProbeAdapterSettings::ChannelAddress;
    using ChannelMapping = ProbeAdapterSettings::ChannelMapping;

    ProbeAdapterImpl(DeviceId deviceId, ProbeAdapterModelId modelId,
                     std::vector<Us4OEM::RawHandle> us4oems,
                     ChannelIdx numberOfChannels,
                     ChannelMapping channelMapping);

    [[nodiscard]] ChannelIdx getNumberOfChannels() const override {
        return numberOfChannels;
    }

    void setTxRxSequence(const std::vector<TxRxParameters> &seq,
                         const ::arrus::ops::us4r::TGCCurve &tgcSamples);

private:
    Logger::Handle logger;
    ProbeAdapterModelId modelId;
    std::vector<Us4OEM::RawHandle> us4oems;
    ChannelIdx numberOfChannels;
    ChannelMapping channelMapping;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERIMPL_H
