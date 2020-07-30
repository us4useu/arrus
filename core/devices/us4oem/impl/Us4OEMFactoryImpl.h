#ifndef ARRUS_CORE_DEVICES_US4OEM_IMPL_US4OEMFACTORYIMPL_H
#define ARRUS_CORE_DEVICES_US4OEM_IMPL_US4OEMFACTORYIMPL_H

#include "core/devices/us4oem/Us4OEMFactory.h"
#include "core/devices/us4oem/impl/Us4OEMImpl.h"
#include "core/devices/us4oem/impl/ius4oem/IUs4OEMFactory.h"

namespace arrus {
class Us4OEMFactoryImpl : public Us4OEMFactory {
public:
    explicit Us4OEMFactoryImpl(IUs4OEMFactory &ius4oemFactory)
            : ius4oemFactory(ius4oemFactory) {}

    Us4OEM::Handle
    getUs4OEM(Ordinal ordinal, const Us4OEMSettings &settings) override {
        IUs4OEMHandle ius4oem = ius4oemFactory.getIUs4OEM(ordinal);

        // Validate settings (whether they can be used to configure the device
        // Initialize the device
        // Set appropriate values

        // Initialize the device
        // Set according to the settings
        // Initialize Us4OEMImpl structures (e.g. the actual ActiveChannelGroups)

        return Us4OEM::Handle(
                new Us4OEMImpl(DeviceId(DeviceType::Us4OEM, ordinal),
                               ius4oem));
    }

private:
    IUs4OEMFactory &ius4oemFactory;
};
}

#endif //ARRUS_CORE_DEVICES_US4OEM_IMPL_US4OEMFACTORYIMPL_H
