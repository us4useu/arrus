#ifndef ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_LNAGAINVALUEMAPPER_H
#define ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_LNAGAINVALUEMAPPER_H

#include <unordered_map>
#include <set>
#include <ius4oem.h>

#include "arrus/core/api/common/types.h"

namespace arrus::devices {

class LNAGainValueMap {

public:
    using ValueType = uint16;

    static LNAGainValueMap &getInstance() {
        static LNAGainValueMap instance;
        return instance;
    }

    us4r::afe58jd18::LNA_GAIN_GBL getEnumValue(const ValueType value) {
        return valueMap.at(value);
    }

    /**
     * Returns a sorted set of available values.
     */
    std::set<ValueType> getAvailableValues() const {
        std::set<ValueType> values;
        std::transform(std::begin(valueMap), std::end(valueMap),
                       std::inserter(values, std::end(values)),
                       [](auto &val) {
                           return val.first;
                       });
        return values;
    }

    LNAGainValueMap(LNAGainValueMap const &) = delete;

    void operator=(LNAGainValueMap const &) = delete;

    LNAGainValueMap(LNAGainValueMap const &&) = delete;

    void operator=(LNAGainValueMap const &&) = delete;

private:
    std::unordered_map<ValueType, us4r::afe58jd18::LNA_GAIN_GBL> valueMap;

    LNAGainValueMap() {
        valueMap.emplace(ValueType(12),
                         us4r::afe58jd18::LNA_GAIN_GBL::LNA_GAIN_GBL_12dB);
        valueMap.emplace(ValueType(18),
                         us4r::afe58jd18::LNA_GAIN_GBL::LNA_GAIN_GBL_18dB);
        valueMap.emplace(ValueType(24),
                         us4r::afe58jd18::LNA_GAIN_GBL::LNA_GAIN_GBL_24dB);
    }

};

}

#endif //ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_LNAGAINVALUEMAPPER_H
