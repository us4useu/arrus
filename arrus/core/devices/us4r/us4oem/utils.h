#ifndef ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_UTILS_H
#define ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_UTILS_H

/** Common functions for Us4OEM handling */
namespace arrus::devices {


/** Converts PRI from the float value [s] to uint32_t value [us] */
inline uint32_t getPRIMicroseconds(float pri) {
    return static_cast<uint32_t>(std::round(pri * 1e6));
}

/** Returns how much to extend the last TX/RX to achieve the given SRI; returns std::nullopt in cas when
  * sri is nullopt.*/
inline std::optional<float> getSRIExtend(
    const std::vector<::arrus::devices::us4r::TxRxParameters>::const_iterator &start,
    const std::vector<::arrus::devices::us4r::TxRxParameters>::const_iterator &end,
    std::optional<float> sri
    ) {
    std::optional<float> lastPriExtend = std::nullopt;
    // Sequence repetition interval.
    if (sri.has_value()) {
        float totalPri = std::accumulate(start, end, 0.0f,
                                         [](const auto &a, const auto &b) {return a + b.getPri();});
        if (totalPri < sri.value()) {
            lastPriExtend = sri.value() - totalPri;
        } else {
            throw IllegalArgumentException(format("Sequence repetition interval {} cannot be set, "
                                                  "sequence total pri is equal {}",
                                                  sri.value(), totalPri));
        }
    }
    return lastPriExtend;
}

inline bool isWaitForSoftMode(arrus::ops::us4r::Scheme::WorkMode workMode) {
    return arrus::ops::us4r::Scheme::isWorkModeManual(workMode) || workMode == ops::us4r::Scheme::WorkMode::HOST;
}


}


#endif//ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_UTILS_H
