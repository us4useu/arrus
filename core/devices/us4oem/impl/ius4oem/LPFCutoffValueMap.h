#ifndef ARRUS_CORE_DEVICES_US4OEM_IMPL_IUS4OEM_LPFCUTOFFVALUEMAP_H
#define ARRUS_CORE_DEVICES_US4OEM_IMPL_IUS4OEM_LPFCUTOFFVALUEMAP_H

#include <unordered_map>
#include <ius4oem.h>

#include "core/common/types.h"

namespace arrus {

class LPFCutoffValueMap {

public:
    static LPFCutoffValueMap &getInstance() {
        static LPFCutoffValueMap instance;
        return instance;
    }

    us4oem::afe58jd18::LPF_PROG getEnumValue(const uint32 value) {
        return valueMap.at(value);
    }

    LPFCutoffValueMap(LPFCutoffValueMap const &) = delete;

    void operator=(LPFCutoffValueMap const &) = delete;

    LPFCutoffValueMap(LPFCutoffValueMap const &&) = delete;

    void operator=(LPFCutoffValueMap const &&) = delete;

private:
    std::unordered_map<uint32, us4oem::afe58jd18::LPF_PROG> valueMap;

    LPFCutoffValueMap() {
        valueMap.emplace(10e6,
                         us4oem::afe58jd18::LPF_PROG::LPF_PROG_10MHz);
        valueMap.emplace(15e6,
                         us4oem::afe58jd18::LPF_PROG::LPF_PROG_15MHz);
        valueMap.emplace(20e6,
                         us4oem::afe58jd18::LPF_PROG::LPF_PROG_20MHz);
        valueMap.emplace(30e6,
                         us4oem::afe58jd18::LPF_PROG::LPF_PROG_30MHz);
        valueMap.emplace(35e6,
                         us4oem::afe58jd18::LPF_PROG::LPF_PROG_35MHz);
        valueMap.emplace(50e6,
                         us4oem::afe58jd18::LPF_PROG::LPF_PROG_50MHz);
    }

};

}

#endif //ARRUS_CORE_DEVICES_US4OEM_IMPL_IUS4OEM_LPFCUTOFFVALUEMAP_H
