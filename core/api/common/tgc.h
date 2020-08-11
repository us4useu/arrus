#ifndef ARRUS_CORE_API_COMMON_TGC_H
#define ARRUS_CORE_API_COMMON_TGC_H

#include <optional>
#include <utility>

#include "arrus/core/api/common/types.h"

namespace arrus {

using TGCSampleValue = float;
using TGCCurve = std::vector<TGCSampleValue>;

class TGCSettings {
public:
    TGCSettings(const std::optional<uint16> &dtgcAttenuation, uint16 pgaGain,
                uint16 lnaGain, TGCCurve tgcSamples)
            : dtgcAttenuation(dtgcAttenuation), pgaGain(pgaGain),
              lnaGain(lnaGain), tgcSamples(std::move(tgcSamples)) {}

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

private:
    std::optional<uint16> dtgcAttenuation;
    uint16 pgaGain;
    uint16 lnaGain;

    TGCCurve tgcSamples;

};

}

#endif //ARRUS_CORE_API_COMMON_TGC_H
