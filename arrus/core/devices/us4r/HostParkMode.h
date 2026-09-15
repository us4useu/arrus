#ifndef ARRUS_CORE_DEVICES_US4R_HOSTPARKMODE_H
#define ARRUS_CORE_DEVICES_US4R_HOSTPARKMODE_H

#include <cstdlib>
#include <cstring>

namespace arrus::devices {

/**
 * ARRUS_HOST_PARK selects how the HOST work mode parks and releases the boards on the Ethernet port.
 *
 * - "element" (unset, DEFAULT): mainline ARRUS v0.14.x behaviour - a WAIT_FOR_SOFT park on the last
 *   entry of EVERY buffer element on EVERY board, released per element by strobing the master, no
 *   HS stop bits. Measured 2026-09-15: two-board 427-511 fps, no handshake machinery.
 * - "last": the 2026-09-13 StreamingTest-parity scheme - one park on the master's last entry,
 *   HS1/HS2 stop bits on, counter-gated resume strobes. Kept for comparison; it produced every
 *   HOST stall measured on 2026-09-14.
 *
 * One reader for the three places that must agree (the sequencer programming in Us4OEMImpl, the
 * stop-bit enable and the release callback in Us4RImpl).
 */
inline bool isHostParkLast() {
    static const bool last = [] {
        const char *v = std::getenv("ARRUS_HOST_PARK");
        return v != nullptr && std::strcmp(v, "last") == 0;
    }();
    return last;
}

}// namespace arrus::devices

#endif//ARRUS_CORE_DEVICES_US4R_HOSTPARKMODE_H
