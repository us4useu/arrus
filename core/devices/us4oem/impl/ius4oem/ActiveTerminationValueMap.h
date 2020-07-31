#ifndef ARRUS_CORE_DEVICES_US4OEM_IMPL_IUS4OEM_ACTIVETERMINATIONVALUEMAP_H
#define ARRUS_CORE_DEVICES_US4OEM_IMPL_IUS4OEM_ACTIVETERMINATIONVALUEMAP_H

#include <unordered_map>
#include <ius4oem.h>

#include "core/common/types.h"

namespace arrus {

class ActiveTerminationValueMap {

public:
    static ActiveTerminationValueMap &getInstance() {
        static ActiveTerminationValueMap instance;
        return instance;
    }

    us4oem::afe58jd18::GBL_ACTIVE_TERM getEnumValue(const uint16 value) {
        return valueMap.at(value);
    }

    ActiveTerminationValueMap(ActiveTerminationValueMap const &) = delete;

    void operator=(ActiveTerminationValueMap const &) = delete;

    ActiveTerminationValueMap(ActiveTerminationValueMap const &&) = delete;

    void operator=(ActiveTerminationValueMap const &&) = delete;

private:
    std::unordered_map<uint16, us4oem::afe58jd18::GBL_ACTIVE_TERM> valueMap;

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
