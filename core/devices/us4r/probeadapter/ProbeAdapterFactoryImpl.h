#ifndef ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERFACTORYIMPL_H
#define ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERFACTORYIMPL_H

#include "arrus/core/api/devices/probe/Probe.h"
#include "ProbeAdapterFactory.h"
#include "arrus/core/devices/us4r/probeadapter/ProbeAdapterSettingsValidator.h"
#include "ProbeAdapterImpl.h"

namespace arrus {

class ProbeAdapterFactoryImpl : public ProbeAdapterFactory {
public:
    ProbeAdapter::Handle
    getProbeAdapter(const ProbeAdapterSettings &settings,
                    std::vector<Us4OEM::RawHandle> us4oems) override {
        const DeviceId id(DeviceType::ProbeAdapter, 0);
        ProbeAdapterSettingsValidator validator(id.getOrdinal());
        validator.validate(settings);
        validator.throwOnErrors();

        assertCorrectNumberOfUs4OEMs(settings, us4oems);

        return ProbeAdapter::Handle(new ProbeAdapterImpl(
                id,
                settings.getModelId(),
                us4oems,
                settings.getNumberOfChannels(),
                settings.getChannelMapping()));
    }

private:
    static void assertCorrectNumberOfUs4OEMs(
            const ProbeAdapterSettings &settings,
            const std::vector<Us4OEM::RawHandle> &us4oems) {
        std::unordered_set<Ordinal> ordinals;
        for(auto value : settings.getChannelMapping()) {
            ordinals.insert(value.first);
        }
        ARRUS_REQUIRES_TRUE(ordinals.size() == us4oems.size(),
                            arrus::format("Incorrect number of us4oems "
                                          "(provided {}, from settings {})",
                                          us4oems.size(), ordinals.size()));
    }
};

}

#endif //ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_PROBEADAPTERFACTORYIMPL_H
