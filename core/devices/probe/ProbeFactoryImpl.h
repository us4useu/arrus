#ifndef ARRUS_CORE_DEVICES_PROBE_PROBEFACTORYIMPL_H
#define ARRUS_CORE_DEVICES_PROBE_PROBEFACTORYIMPL_H

#include <memory>

#include "arrus/core/devices/probe/ProbeSettingsValidator.h"
#include "arrus/core/api/devices/Device.h"
#include "arrus/core/api/devices/us4r/ProbeAdapter.h"
#include "arrus/common/asserts.h"
#include "arrus/common/format.h"
#include "arrus/core/api/common/exceptions.h"
#include "arrus/core/devices/probe/ProbeFactory.h"
#include "ProbeImpl.h"

namespace arrus::devices {

class ProbeFactoryImpl : public ProbeFactory {
public:
    ProbeImplBase::Handle getProbe(const ProbeSettings &settings,
                           ProbeAdapterImplBase::RawHandle adapter) override {
        DeviceId id(DeviceType::Probe, 0);
        ProbeSettingsValidator validator(id.getOrdinal());
        validator.validate(settings);
        validator.throwOnErrors();

        // Additionally, verify destination channels (should be in range
        // available for the given adapter).
        for(auto value : settings.getChannelMapping()) {
            ARRUS_REQUIRES_IN_CLOSED_INTERVAL(
                    value, 0, adapter->getNumberOfChannels(),
                    ::arrus::format("Destination channel address: {} "
                                  "exceeds maximum number of channels ({})"
                                  " of the underlying probe adapter.",
                                  value, adapter->getNumberOfChannels())
            );
        }
        return std::make_unique<ProbeImpl>(id, settings.getModel(), adapter,
                                       settings.getChannelMapping());
    }
};

}

#endif //ARRUS_CORE_DEVICES_PROBE_PROBEFACTORYIMPL_H
