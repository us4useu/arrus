#ifndef ARRUS_CORE_API_DEVICES_US4R_US4OEMINTERRUPT_H
#define ARRUS_CORE_API_DEVICES_US4R_US4OEMINTERRUPT_H

#include <array>
#include <functional>
#include <unordered_map>

#include "arrus/core/api/devices/DeviceId.h"

namespace arrus::devices {

/**
 * The set of us4OEM interrupts a user callback can be registered for.
 */
enum class Us4OEMInterrupt {
    PROBE_NOT_CONNECTED,
    PULSER_INTERRUPT,
    TX_TIMEOUT,
    WATCHDOG_IRQ0,
    WATCHDOG_IRQ1,
    HVPS_FUSE,
};

/**
 * Interrupts that, when raised, indicate the system has entered (or should
 * enter) a safe state. Includes every supported us4OEM interrupt except
 * WATCHDOG_IRQ0, which is only an early warning.
 */
inline constexpr std::array<Us4OEMInterrupt, 5> SAFE_STATE_INTERRUPTS{
    Us4OEMInterrupt::PROBE_NOT_CONNECTED,
    Us4OEMInterrupt::PULSER_INTERRUPT,
    Us4OEMInterrupt::TX_TIMEOUT,
    Us4OEMInterrupt::WATCHDOG_IRQ1,
    Us4OEMInterrupt::HVPS_FUSE,
};

/**
 * User-provided callback invoked when the registered us4OEM interrupt occurs.
 *
 * The interrupt that triggered the callback is identified by the key under
 * which the callback was registered in the Us4OEMInterruptCallbacksMap, so
 * the callback only needs the ordinal number of the OEM that raised it.
 *
 * @param oem the ordinal number of the OEM that raised the interrupt
 */
using Us4OEMInterruptCallback = std::function<void(Ordinal oem)>;

/**
 * Map from us4OEM system interrupt to the user-provided callback to invoke
 * when that interrupt occurs. Callbacks are registered only for the keys
 * that are present in the map.
 */
using Us4OEMInterruptCallbacksMap = std::unordered_map<Us4OEMInterrupt, Us4OEMInterruptCallback>;

}

#endif //ARRUS_CORE_API_DEVICES_US4R_US4OEMINTERRUPT_H
