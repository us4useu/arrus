#ifndef ARRUS_CORE_DEVICES_US4R_RXSETTINGS_H
#define ARRUS_CORE_DEVICES_US4R_RXSETTINGS_H

#include "arrus/core/api/devices/us4r/RxSettings.h"

namespace arrus::devices {

std::ostream &operator<<(std::ostream &os, const RxSettings &settings);

class RxSettingsBuilder {
public:
    using TGCCurve = arrus::ops::us4r::TGCCurve;

    explicit RxSettingsBuilder(const RxSettings& rxSettings) {
        dtgcAttenuation = rxSettings.getDtgcAttenuation();
        pgaGain = rxSettings.getPgaGain();
        lnaGain = rxSettings.getLnaGain();
        tgcSamples = rxSettings.getTgcSamples();
        lpfCutoff = rxSettings.getLpfCutoff();
        activeTermination = rxSettings.getActiveTermination();
        applyTgcCharacteristic = rxSettings.isApplyTgcCharacteristic();
    }

    RxSettingsBuilder *setDtgcAttenuation(const std::optional<uint16> &value) {
        RxSettingsBuilder::dtgcAttenuation = value;
        return this;
    }

    RxSettingsBuilder *setPgaGain(uint16 value) {
        RxSettingsBuilder::pgaGain = value;
        return this;
    }

    RxSettingsBuilder *setLnaGain(uint16 value) {
        RxSettingsBuilder::lnaGain = value;
        return this;
    }

    RxSettingsBuilder *setTgcSamples(const TGCCurve &value) {
        RxSettingsBuilder::tgcSamples = value;
        return this;
    }

    RxSettingsBuilder *setLpfCutoff(uint32 value) {
        RxSettingsBuilder::lpfCutoff = value;
        return this;
    }

    RxSettingsBuilder *setActiveTermination(const std::optional<uint16> &value) {
        RxSettingsBuilder::activeTermination = value;
        return this;
    }

    RxSettingsBuilder *setApplyTgcCharacteristic(bool value) {
        RxSettingsBuilder::applyTgcCharacteristic = value;
        return this;
    }

    RxSettings build() {
        return RxSettings{dtgcAttenuation, pgaGain, lnaGain, tgcSamples, lpfCutoff,
                          activeTermination, applyTgcCharacteristic};
    }

private:
    std::optional<uint16> dtgcAttenuation;
    uint16 pgaGain;
    uint16 lnaGain;
    TGCCurve tgcSamples;
    uint32 lpfCutoff;
    std::optional<uint16> activeTermination;
    bool applyTgcCharacteristic;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_RXSETTINGS_H
