#ifndef ARRUS_CORE_DEVICES_US4OEM_IMPL_IUS4OEM_ACTIVETERMINATIONVALUEMAP_H
#define ARRUS_CORE_DEVICES_US4OEM_IMPL_IUS4OEM_ACTIVETERMINATIONVALUEMAP_H

#include <unordered_map>
#include <set>
#include <ius4oem.h>

#include "core/common/types.h"

namespace arrus {

class ActiveTerminationValueMap {

public:
    using ActiveTerminationValueType = uint16;

    static ActiveTerminationValueMap &getInstance() {
        static ActiveTerminationValueMap instance;
        return instance;
    }

    us4oem::afe58jd18::GBL_ACTIVE_TERM
    getEnumValue(const ActiveTerminationValueType value) {
        return valueMap.at(value);
    }

    /**
     * Returns a sorted set of available values.
     */
    std::set<ActiveTerminationValueType> getAvailableValues() const {
        std::set<ActiveTerminationValueType> values;
        std::transform(std::begin(valueMap), std::end(valueMap),
                       std::back_inserter(values),
                       [](auto &val) {
                           val.first;
                       });
        return values;
    }

    ActiveTerminationValueMap(ActiveTerminationValueMap const &) = delete;

    void operator=(ActiveTerminationValueMap const &) = delete;

    ActiveTerminationValueMap(ActiveTerminationValueMap const &&) = delete;

    void operator=(ActiveTerminationValueMap const &&) = delete;

private:
    std::unordered_map<ActiveTerminationValueType, us4oem::afe58jd18::GBL_ACTIVE_TERM> valueMap;

    ActiveTerminationValueMap() {
        valueMap.emplace(50,
                         us4oem::afe58jd18::GBL_ACTIVE_TERM::GBL_ACTIVE_TERM_50);
        valueMap.emplace(100,
                         us4oem::afe58jd18::GBL_ACTIVE_TERM::GBL_ACTIVE_TERM_100);

        valueMap.emplace(200,
                         us4oem::afe58jd18::GBL_ACTIVE_TERM::GBL_ACTIVE_TERM_200);
        valueMap.emplace(400,
                         us4oem::afe58jd18::GBL_ACTIVE_TERM::GBL_ACTIVE_TERM_400);
    }

};

}
#endif //ARRUS_CORE_DEVICES_US4OEM_IMPL_IUS4OEM_ACTIVETERMINATIONVALUEMAP_H
