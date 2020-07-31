#ifndef ARRUS_CORE_DEVICES_US4OEM_IMPL_MAPPERS_LNAGAINVALUEMAPPER_H
#define ARRUS_CORE_DEVICES_US4OEM_IMPL_MAPPERS_LNAGAINVALUEMAPPER_H

#include <unordered_map>
#include <ius4oem.h>

#include "core/common/types.h"

namespace arrus {

class PGAGainValueMap {

public:
    static PGAGainValueMap &getInstance() {
        static PGAGainValueMap instance;
        return instance;
    }

    us4oem::afe58jd18::PGA_GAIN getEnumValue(const uint8 value) {
        return valueMap.at(value);
    }

    PGAGainValueMap(PGAGainValueMap const &) = delete;

    void operator=(PGAGainValueMap const &) = delete;

    PGAGainValueMap(PGAGainValueMap const &&) = delete;

    void operator=(PGAGainValueMap const &&) = delete;

private:
    std::unordered_map<uint8, us4oem::afe58jd18::PGA_GAIN> valueMap;

    PGAGainValueMap() {
        valueMap.emplace(24, us4oem::afe58jd18::PGA_GAIN::PGA_GAIN_24dB);
        valueMap.emplace(30, us4oem::afe58jd18::PGA_GAIN::PGA_GAIN_30dB);
    }

};

}

#endif //ARRUS_CORE_DEVICES_US4OEM_IMPL_MAPPERS_LNAGAINVALUEMAPPER_H
