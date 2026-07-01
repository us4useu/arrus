#ifndef ARRUS_CORE_API_DEVICES_US4R_IO_CAPABILITY_H
#define ARRUS_CORE_API_DEVICES_US4R_IO_CAPABILITY_H

namespace arrus::devices::us4r {

/**
 * Us4R IO capability.
 */
enum class IOCapability {
    /**
     * Probe-connected check capability. This capability configures us4OEMs to react (stop the system, turn off HV) when
     * there is no connection between the us4R system and the dedicated probe pin ("the probe is not connected").
     */
    PROBE_CONNECTED_CHECK,
    /**
     * Frame metadata capability (e.g. external signal encoder in frame metadata).
     * NOTE: this capability requires a custom firmware development, dedicated to the target application.
     */
    FRAME_METADATA,
};

}

#endif//ARRUS_CORE_API_DEVICES_US4R_IO_CAPABILITY_H
