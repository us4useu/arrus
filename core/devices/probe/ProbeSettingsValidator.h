#ifndef ARRUS_CORE_DEVICES_PROBE_PROBESETTINGSVALIDATOR_H
#define ARRUS_CORE_DEVICES_PROBE_PROBESETTINGSVALIDATOR_H

#include "arrus/core/api/devices/probe/ProbeSettings.h"
#include "arrus/core/devices/DeviceSettingsValidator.h"

namespace arrus {

class ProbeSettingsValidator : public DeviceSettingsValidator<ProbeSettings> {

public:
    explicit ProbeSettingsValidator(const Ordinal ordinal)
    : DeviceSettingsValidator(DeviceId(DeviceType::Probe, ordinal)) {}

    void validate(const ProbeSettings &obj) override {

        expectAtMost("numberOfElements",
                  obj.getModel().getNumberOfElements().size(), (size_t)2,
                  "(size)");
        expectEqual("numberOfElements",
                    obj.getModel().getNumberOfElements().size(),
                    obj.getModel().getPitch().size(),
                    " (size, comparing with pitch)");
        // Validating model.
        expectAllPositive("pitch", obj.getModel().getPitch().getValues());
        expectAllPositive<double>(
                "txFrequencyRange",
                {obj.getModel().getTxFrequencyRange().left(),
                 obj.getModel().getTxFrequencyRange().right()});
        expectAllInRange(
                "numberOfElements",
                obj.getModel().getNumberOfElements().getValues(),
                (ProbeModel::ElementIdxType)1,
                std::numeric_limits<ProbeModel::ElementIdxType>::max()
        );
        expectEqual(
                "nElements",
                obj.getChannelMapping().size(),
                (size_t)obj.getModel().getNumberOfElements().product()
        );
        expectUnique("channelMapping", obj.getChannelMapping());
    }
};

}

#endif //ARRUS_CORE_DEVICES_PROBE_PROBESETTINGSVALIDATOR_H
