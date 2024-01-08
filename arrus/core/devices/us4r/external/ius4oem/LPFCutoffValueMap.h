#ifndef ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_LPFCUTOFFVALUEMAP_H
#define ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_LPFCUTOFFVALUEMAP_H

#include <unordered_map>
#include <set>
#include <ius4oem.h>

#include "arrus/core/api/common/types.h"

namespace arrus::devices {

class LPFCutoffValueMap {

public:
    using LPFCutoffValueType = uint32;

    static LPFCutoffValueMap &getInstance() {
        static LPFCutoffValueMap instance;
        return instance;
    }

    ::us4r::afe58jd48::LPF_PROG getEnumValue(const LPFCutoffValueType value) {
        return valueMap.at(value);
    }

    /**
     * Returns a sorted set of available values.
     */
    std::set<LPFCutoffValueType> getAvailableValues() const {
        std::set<LPFCutoffValueType> values;
        std::transform(std::begin(valueMap), std::end(valueMap),
                       std::inserter(values, std::end(values)),
                       [](auto &val) {
                           return val.first;
                       });
        return values;
    }

    LPFCutoffValueMap(LPFCutoffValueMap const &) = delete;

    void operator=(LPFCutoffValueMap const &) = delete;

    LPFCutoffValueMap(LPFCutoffValueMap const &&) = delete;

    void operator=(LPFCutoffValueMap const &&) = delete;

private:
    std::unordered_map<LPFCutoffValueType, ::us4r::afe58jd48::LPF_PROG> valueMap;

    LPFCutoffValueMap() {
        valueMap.emplace(10000000,
                         ::us4r::afe58jd48::LPF_PROG::LPF_PROG_10MHz);
        valueMap.emplace(15000000,
                         ::us4r::afe58jd48::LPF_PROG::LPF_PROG_15MHz);
        valueMap.emplace(20000000,
                         ::us4r::afe58jd48::LPF_PROG::LPF_PROG_20MHz);
        valueMap.emplace(30000000,
                         ::us4r::afe58jd48::LPF_PROG::LPF_PROG_30MHz);
        valueMap.emplace(40000000,
                         ::us4r::afe58jd48::LPF_PROG::LPF_PROG_40MHz);
        valueMap.emplace(50000000,
                         ::us4r::afe58jd48::LPF_PROG::LPF_PROG_50MHz);
        valueMap.emplace(60000000,
                         ::us4r::afe58jd48::LPF_PROG::LPF_PROG_60MHz);
    }

};

}

#endif //ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_LPFCUTOFFVALUEMAP_H
