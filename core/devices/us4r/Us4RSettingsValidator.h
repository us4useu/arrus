#ifndef ARRUS_CORE_DEVICES_US4R_US4RSETTINGSVALIDATOR_H
#define ARRUS_CORE_DEVICES_US4R_US4RSETTINGSVALIDATOR_H

#include "arrus/core/api/devices/us4r/Us4RSettings.h"
#include "arrus/core/devices/DeviceSettingsValidator.h"

namespace arrus {
class Us4RSettingsValidator : public DeviceSettingsValidator<Us4RSettings> {
public:
    explicit Us4RSettingsValidator(Ordinal moduleOrdinal)
            : DeviceSettingsValidator<Us4RSettings>(
            DeviceId(DeviceType::Us4R, moduleOrdinal)) {}

    void validate(const Us4RSettings &obj) override {
        if(obj.getUs4OEMSettings().empty()) {
            expectTrue("probe adapter settings",
                       obj.getProbeAdapterSettings().has_value(),
                       "At least probe adapter settings is required.");
            // TODO wymagaj tez podpietej glowicy
            expectTrue("tgc settings",
                       obj.getRxSettings().has_value(),
                       "Us4R TGC settings must be provided.");
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
        }
        // The exact TGC settings should be verified by the underlying Us4OEMs.
    }

};
}

#endif //ARRUS_CORE_DEVICES_US4R_US4RSETTINGSVALIDATOR_H
