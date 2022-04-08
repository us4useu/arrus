#ifndef ARRUS_CORE_DEVICES_US4R_US4RSETTINGSVALIDATOR_H
#define ARRUS_CORE_DEVICES_US4R_US4RSETTINGSVALIDATOR_H

#include "arrus/core/api/devices/us4r/Us4RSettings.h"
#include "arrus/core/devices/SettingsValidator.h"

namespace arrus::devices {
class Us4RSettingsValidator : public SettingsValidator<Us4RSettings> {
public:
    explicit Us4RSettingsValidator(Ordinal moduleOrdinal)
            : SettingsValidator<Us4RSettings>(
            DeviceId(DeviceType::Us4R, moduleOrdinal)) {}

    void validate(const Us4RSettings &obj) override {
        if(obj.getUs4OEMSettings().empty()) {
            expectTrue("probe adapter settings",
                       obj.getProbeAdapterSettings().has_value(),
                       "Probe adapter settings are required.");
            expectTrue("probe settings",
                       obj.getProbeSettings().has_value(),
                       "Probe settings are required.");
            expectTrue("tgc settings",
                       obj.getRxSettings().has_value(),
                       "Us4R TGC settings must be provided.");
            if(hasErrors()) {
                return;
            }
            auto &adapterMapping = obj.getProbeAdapterSettings()->getChannelMapping();
            Ordinal nus4oemsAdapter = std::max_element(
                                   std::begin(adapterMapping), std::end(adapterMapping),
                                   [] (auto &a, auto &b) {return a.first < b.first;} )->first;
            nus4oemsAdapter += 1; // The above starts from 0
            if(obj.getNumberOfUs4oems().has_value()) {
                expectTrue(
                    "nus4oems",
                    obj.getNumberOfUs4oems() >= nus4oemsAdapter,
                    "The declared number of us4oems should not be less than the number of us4oems used in the probe "
                    "adapter channel mapping"
                );
            }
            if(! obj.getAdapterToUs4RModuleNumber().empty()) {
                expectTrue(
                    "adapter to us4R module number",
                    obj.getAdapterToUs4RModuleNumber().size() == nus4oemsAdapter,
                    "The size of the mapping should be equal to the number of us4oems used in the probe adapter "
                    "channel mapping"
                );
            }
        } else {
            expectFalse(
                    "probe adapter settings",
                    obj.getProbeAdapterSettings().has_value(),
                    "Probe settings should not be set "
                    "(at least one custom Us4OEM setting was detected).");

            expectFalse(
                    "probe settings",
                    obj.getProbeSettings().has_value(),
                    "Probe settings should not be set "
                    "(at least one custom Us4OEM setting was detected).");
            if(obj.getNumberOfUs4oems().has_value()) {
                expectTrue(
                    "us4r settings",
                    obj.getNumberOfUs4oems().value() == obj.getUs4OEMSettings().size(),
                    "The declared number of us4OEMs should be equal to the number of provided Us4OEM settings"
                );
            }
        }
        // The exact TGC settings should be verified by the underlying Us4OEMs.
    }

};
}

#endif //ARRUS_CORE_DEVICES_US4R_US4RSETTINGSVALIDATOR_H
