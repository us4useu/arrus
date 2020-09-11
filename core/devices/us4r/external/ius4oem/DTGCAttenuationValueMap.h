#ifndef ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_DTGCATTENUATIONVALUEMAP_H
#define ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_DTGCATTENUATIONVALUEMAP_H

#include <unordered_map>
#include <set>
#include <ius4oem.h>

#include "arrus/core/api/common/types.h"

namespace arrus::devices {

class DTGCAttenuationValueMap {

public:
    using ValueType = uint16;

    static DTGCAttenuationValueMap &getInstance() {
        static DTGCAttenuationValueMap instance;
        return instance;
    }

    us4r::afe58jd18::DIG_TGC_ATTENUATION
    getEnumValue(const ValueType value) {
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

    DTGCAttenuationValueMap(DTGCAttenuationValueMap const &) = delete;

    void operator=(DTGCAttenuationValueMap const &) = delete;

    DTGCAttenuationValueMap(DTGCAttenuationValueMap const &&) = delete;

    void operator=(DTGCAttenuationValueMap const &&) = delete;

private:
    std::unordered_map<ValueType, us4r::afe58jd18::DIG_TGC_ATTENUATION> valueMap;

    DTGCAttenuationValueMap() {
        valueMap.emplace(ValueType(0),
                         us4r::afe58jd18::DIG_TGC_ATTENUATION::DIG_TGC_ATTENUATION_0dB);
        valueMap.emplace(ValueType(6),
                         us4r::afe58jd18::DIG_TGC_ATTENUATION::DIG_TGC_ATTENUATION_6dB);
        valueMap.emplace(ValueType(12),
                         us4r::afe58jd18::DIG_TGC_ATTENUATION::DIG_TGC_ATTENUATION_12dB);
        valueMap.emplace(ValueType(18),
                         us4r::afe58jd18::DIG_TGC_ATTENUATION::DIG_TGC_ATTENUATION_18dB);
        valueMap.emplace(ValueType(24),
                         us4r::afe58jd18::DIG_TGC_ATTENUATION::DIG_TGC_ATTENUATION_24dB);
        valueMap.emplace(ValueType(30),
                         us4r::afe58jd18::DIG_TGC_ATTENUATION::DIG_TGC_ATTENUATION_30dB);
        valueMap.emplace(ValueType(36),
                         us4r::afe58jd18::DIG_TGC_ATTENUATION::DIG_TGC_ATTENUATION_36dB);
        valueMap.emplace(ValueType(42),
                         us4r::afe58jd18::DIG_TGC_ATTENUATION::DIG_TGC_ATTENUATION_42dB);
    }


};

}

#endif //ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_DTGCATTENUATIONVALUEMAP_H
