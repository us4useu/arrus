#ifndef ARRUS_CORE_DEVICES_US4R_TGCSETTINGS_H
#define ARRUS_CORE_DEVICES_US4R_TGCSETTINGS_H

#include <utility>
#include <optional>

#include "arrus/core/api/common/types.h"
#include "arrus/core/api/ops/us4r/tgc.h"

namespace arrus::devices {

class TgcSettings {
public:
    TgcSettings(ops::us4r::TGCCurve tgcCurve,
                uint8 lnaGain,
                uint8 pgaGain,
                uint8 dtgcAttenuation,
                bool applyCharacteristic)
            : tgcCurve(std::move(tgcCurve)),
              lnaGain(lnaGain),
              pgaGain(pgaGain),
              dtgcAttenuation(dtgcAttenuation),
              applyCharacteristic(applyCharacteristic) {}

    [[nodiscard]] const ::arrus::ops::us4r::TGCCurve &getTgcCurve() const {
        return tgcCurve;
    }
    [[nodiscard]] uint8 getLnaGain() const {
        return lnaGain;
    }
    [[nodiscard]] uint8 getPgaGain() const {
        return pgaGain;
    }
    [[nodiscard]] bool isApplyCharacteristic() const {
        return applyCharacteristic;
    }

    [[nodiscard]] std::optional<uint8> getDtgcAttenuation() const {
        return dtgcAttenuation;
    }

private:
    ::arrus::ops::us4r::TGCCurve tgcCurve;
    uint8 lnaGain;
    uint8 pgaGain;
    std::optional<uint8> dtgcAttenuation;
    bool applyCharacteristic;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_TGCSETTINGS_H
