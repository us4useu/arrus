#ifndef ARRUS_CORE_DEVICES_US4R_US4OEM_TXWAVEFORMCONVERTER_H
#define ARRUS_CORE_DEVICES_US4R_US4OEM_TXWAVEFORMCONVERTER_H

#include "arrus/core/api/ops/us4r/Waveform.h"
#include <vector>

namespace arrus::devices {

/**
 * Converts tx waveform from the HAL waveform description to the STHV 1600 pulser memory definition.
 */
class TxWaveformConverter {
public:

    static std::vector<uint32_t> toPulser(const ::arrus::ops::us4r::Waveform &wf) {
        std::vector<uint32_t> result;
        return result;
    }
private:
    uint32_t setRepeatType(uint32_t input, uint32_t type) {
        return input;
    }

    uint32_t set

    uint32_t setBitField(uint32_t input, uint32_t offset, uint32_t value) {
        value = value << offset;
        return input &
    }
};

}

#endif//ARRUS_CORE_DEVICES_US4R_US4OEM_TXWAVEFORMCONVERTER_H
