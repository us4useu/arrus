#ifndef ARRUS_CORE_DEVICES_US4OEM_IMPL_MAPPERS_LNAGAINVALUEMAPPER_H
#define ARRUS_CORE_DEVICES_US4OEM_IMPL_MAPPERS_LNAGAINVALUEMAPPER_H

#include <unordered_map>
#include <set>
#include <ius4oem.h>

#include "core/common/types.h"

namespace arrus {

class LNAGainValueMap {

public:
    using LNAGainValueType = uint8;

    static LNAGainValueMap &getInstance() {
        static LNAGainValueMap instance;
        return instance;
    }

    us4oem::afe58jd18::LNA_GAIN_GBL getEnumValue(const LNAGainValueType value) {
        return valueMap.at(value);
    }

    /**
     * Returns a sorted set of available values.
     */
    std::set<LNAGainValueType> getAvailableValues() const {
        std::set<LNAGainValueType> values;
        std::transform(std::begin(valueMap), std::end(valueMap),
                       std::back_inserter(values),
                       [](auto &val) {
                           val.first;
                       });
        return values;
    }

    LNAGainValueMap(LNAGainValueMap const &) = delete;

    void operator=(LNAGainValueMap const &) = delete;

    LNAGainValueMap(LNAGainValueMap const &&) = delete;

    void operator=(LNAGainValueMap const &&) = delete;

private:
    std::unordered_map<LNAGainValueType, us4oem::afe58jd18::LNA_GAIN_GBL> valueMap;

    LNAGainValueMap() {
        valueMap.emplace(12,
                         us4oem::afe58jd18::LNA_GAIN_GBL::LNA_GAIN_GBL_12dB);
        valueMap.emplace(18,
                         us4oem::afe58jd18::LNA_GAIN_GBL::LNA_GAIN_GBL_18dB);
        valueMap.emplace(24,
                         us4oem::afe58jd18::LNA_GAIN_GBL::LNA_GAIN_GBL_24dB);
    }

};

}

#endif //ARRUS_CORE_DEVICES_US4OEM_IMPL_MAPPERS_LNAGAINVALUEMAPPER_H
