#ifndef ARRUS_CORE_API_DEVICES_US4R_RXSETTINGS_H
#define ARRUS_CORE_API_DEVICES_US4R_RXSETTINGS_H

#include <optional>
#include <utility>
#include <ostream>

#include "arrus/core/api/common/types.h"
#include "arrus/core/api/ops/us4r/tgc.h"

namespace arrus::devices {

class RxSettings {
public:
    using TGCSample = arrus::ops::us4r::TGCSampleValue;
    using TGCCurve = arrus::ops::us4r::TGCCurve;

    RxSettings(
            const std::optional<uint16> &dtgcAttenuation, uint16 pgaGain,
            uint16 lnaGain, TGCCurve tgcSamples, uint32 lpfCutoff,
            const std::optional<uint16> &activeTermination)
            : dtgcAttenuation(dtgcAttenuation), pgaGain(pgaGain),
              lnaGain(lnaGain), tgcSamples(std::move(tgcSamples)),
              lpfCutoff(lpfCutoff), activeTermination(activeTermination) {}

    [[nodiscard]] const std::optional<uint16> &getDTGCAttenuation() const {
        return dtgcAttenuation;
    }

    [[nodiscard]] uint16 getPGAGain() const {
        return pgaGain;
    }

    [[nodiscard]] uint16 getLNAGain() const {
        return lnaGain;
    }

    [[nodiscard]] const TGCCurve &getTGCSamples() const {
        return tgcSamples;
    }

    [[nodiscard]] uint32 getLPFCutoff() const {
        return lpfCutoff;
    }

    [[nodiscard]] const std::optional<uint16> &getActiveTermination() const {
        return activeTermination;
    }



private:
    std::optional<uint16> dtgcAttenuation;
    uint16 pgaGain;
    uint16 lnaGain;

    TGCCurve tgcSamples;
    uint32 lpfCutoff;
    std::optional<uint16> activeTermination;
};

}

#endif //ARRUS_CORE_API_DEVICES_US4R_RXSETTINGS_H
