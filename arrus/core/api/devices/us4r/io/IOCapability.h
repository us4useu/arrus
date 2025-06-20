#ifndef ARRUS_CORE_API_DEVICES_US4R_IO_CAPABILITY_H
#define ARRUS_CORE_API_DEVICES_US4R_IO_CAPABILITY_H

namespace arrus::devices::us4r {

enum class IOCapability {
    PROBE_CONNECTED_CHECK,
    FRAME_METADATA, // To be available in v0.9.0
};

}

#endif//ARRUS_CORE_API_DEVICES_US4R_IO_CAPABILITY_H
