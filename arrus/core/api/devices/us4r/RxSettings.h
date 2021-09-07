#ifndef ARRUS_CORE_API_DEVICES_US4R_RXSETTINGS_H
#define ARRUS_CORE_API_DEVICES_US4R_RXSETTINGS_H

#include <optional>
#include <utility>
#include <ostream>

#include "arrus/core/api/common/types.h"
#include "arrus/core/api/ops/us4r/tgc.h"

namespace arrus::devices {

/*
 * AFE (RX) settings.
 *
 * Us4R AFE settings currently includes:
 * - DTGC attenuation, available: 0, 6, 12, 18, 24, 30, 36, 42 [dB] or std::nullopt; nullopt turns off DTGC.
 * - LNA gain, available: 12, 18, 24 [dB].
 * - PGA gain, available: 24, 30 [dB].
 * - TGC samples: analog TGC curve samples [dB]. Up to 1022 samples. TGC samples should be in range
 *   [min, max](closed interval) where min = (lna gain + pga gain)-40, and max = (lna gain, pga gain).
 *   Empty list turns off analog TGC.
 * - Active termination, available: 50, 100, 200, 400 [Ohm] or std::nullopt. null opt turns off active termination
 * - LPF cutoff, available: 10000000, 15000000, 20000000, 30000000, 35000000, 50000000 [Hz].
 * - applyTgcCharacteristic: whether to apply pre-computed (by us4us) TGC response characteristic, so that the observed
 *   gain better corresponds to the applied one.
 *
 * Constraints:
 * - only one of the following can be turned on: DTGC or analog TGC.
 * - when applyTgcCharacteristic == true, LNA and PGA gain has to be (current limitation of the selected TGC
 *   characteristic).
 */
class RxSettings {
public:
    using TGCSample = arrus::ops::us4r::TGCSampleValue;
    using TGCCurve = arrus::ops::us4r::TGCCurve;
    static constexpr float TGC_ATTENUATION_RANGE = 40.0f;

    /**
     * A helper function that computes a pair of (min, max) acceptable analog TGC gain for given PGA and LNA gain
     * values.
     *
     * @param pgaGain PGA gain value to consider
     * @param lnaGain LNA gain value to consider
     * @return a pair (min, max) acceptable sample value.
     */
    static std::pair<float, float> getTgcMinMax(uint16 pgaGain, uint16 lnaGain) {
        float max = float(pgaGain) + float(lnaGain);
        float min = max-TGC_ATTENUATION_RANGE;
        return {min, max};
    }

    RxSettings(const std::optional<uint16> &dtgcAttenuation, uint16 pgaGain, uint16 lnaGain,
               TGCCurve tgcSamples, uint32 lpfCutoff, const std::optional<uint16> &activeTermination,
               bool applyTgcCharacteristic = true)
            : dtgcAttenuation(dtgcAttenuation), pgaGain(pgaGain), lnaGain(lnaGain), tgcSamples(std::move(tgcSamples)),
              lpfCutoff(lpfCutoff), activeTermination(activeTermination),
              applyTgcCharacteristic(applyTgcCharacteristic) {}


    const std::optional<uint16> &getDtgcAttenuation() const {
        return dtgcAttenuation;
    }

    uint16 getPgaGain() const {
        return pgaGain;
    }

    uint16 getLnaGain() const {
        return lnaGain;
    }

    const TGCCurve &getTgcSamples() const {
        return tgcSamples;
    }

    uint32 getLpfCutoff() const {
        return lpfCutoff;
    }

    const std::optional<uint16> &getActiveTermination() const {
        return activeTermination;
    }

    bool isApplyTgcCharacteristic() const {
        return applyTgcCharacteristic;
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

#endif //ARRUS_CORE_API_DEVICES_US4R_RXSETTINGS_H
