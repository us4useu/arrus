#ifndef ARRUS_CORE_DEVICES_US4OEM_IMPL_US4OEMFACTORYIMPL_H
#define ARRUS_CORE_DEVICES_US4OEM_IMPL_US4OEMFACTORYIMPL_H

#include "arrus/core/devices/us4oem/Us4OEMFactory.h"
#include "arrus/core/devices/us4oem/impl/Us4OEMImpl.h"
#include "arrus/core/devices/us4oem/impl/Us4OEMSettingsValidator.h"
#include "arrus/core/external/ius4oem/IUs4OEMFactory.h"

namespace arrus {
class Us4OEMFactoryImpl : public Us4OEMFactory {
public:
    explicit Us4OEMFactoryImpl(IUs4OEMFactory &ius4oemFactory)
            : ius4oemFactory(ius4oemFactory) {}

    Us4OEM::Handle
    getUs4OEM(Ordinal ordinal, const IUs4OEMHandle &ius4oem,
              const Us4OEMSettings &settings) override {

        // Validate settings.
        Us4OEMSettingsValidator validator(settings);
        validator.throwOnErrors();

        // Us4OEM Initial configuration.
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
