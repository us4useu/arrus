#ifndef ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_ACTIVETERMINATIONVALUEMAP_H
#define ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_ACTIVETERMINATIONVALUEMAP_H

#include <unordered_map>
#include <set>
#include <ius4oem.h>

#include "arrus/core/api/common/types.h"

namespace arrus::devices {

class ActiveTerminationValueMap {

public:
    using ValueType = uint16;

    static ActiveTerminationValueMap &getInstance() {
        static ActiveTerminationValueMap instance;
        return instance;
    }

    ::us4r::afe58jd48::GBL_ACTIVE_TERM getEnumValue(const ValueType value) {
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

    ActiveTerminationValueMap(ActiveTerminationValueMap const &) = delete;

    void operator=(ActiveTerminationValueMap const &) = delete;

    ActiveTerminationValueMap(ActiveTerminationValueMap const &&) = delete;

    void operator=(ActiveTerminationValueMap const &&) = delete;

private:
    std::unordered_map<ValueType, ::us4r::afe58jd48::GBL_ACTIVE_TERM> valueMap{};

    ActiveTerminationValueMap() {
        valueMap.emplace((ValueType)50,
                         ::us4r::afe58jd48::GBL_ACTIVE_TERM::GBL_ACTIVE_TERM_50);
        valueMap.emplace((ValueType)100,
                         ::us4r::afe58jd48::GBL_ACTIVE_TERM::GBL_ACTIVE_TERM_100);

        valueMap.emplace((ValueType)200,
                         ::us4r::afe58jd48::GBL_ACTIVE_TERM::GBL_ACTIVE_TERM_200);
        valueMap.emplace((ValueType)400,
                         ::us4r::afe58jd48::GBL_ACTIVE_TERM::GBL_ACTIVE_TERM_400);
    }

};

}
#endif //ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_ACTIVETERMINATIONVALUEMAP_H
