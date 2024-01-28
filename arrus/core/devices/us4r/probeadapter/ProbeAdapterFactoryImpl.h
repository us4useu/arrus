#ifndef ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERFACTORYIMPL_H
#define ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERFACTORYIMPL_H

#include <memory>

#include "arrus/core/api/devices/probe/Probe.h"
#include "ProbeAdapterFactory.h"
#include "arrus/core/devices/us4r/probeadapter/ProbeAdapterSettingsValidator.h"
#include "ProbeAdapterImpl.h"

namespace arrus::devices {

class ProbeAdapterFactoryImpl : public ProbeAdapterFactory {
public:
    ProbeAdapterImplBase::Handle
    getProbeAdapter(const ProbeAdapterSettings &settings,
                    const std::vector<Us4OEMImplBase::RawHandle> &us4oems) override {
        const DeviceId id(DeviceType::ProbeAdapter, 0);


        return std::make_unique<ProbeAdapterImpl>(
            id,
            settings.getModelId(),
            us4oems,
            settings.getNumberOfChannels(),
            settings.getChannelMapping(),
            settings.getIOSettings()
        );
    }
};

}

#endif //ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERFACTORYIMPL_H
