#ifndef ARRUS_CORE_API_DEVICES_ULTRASOUNDDEVICE_H
#define ARRUS_CORE_API_DEVICES_ULTRASOUNDDEVICE_H

#include "arrus/core/devices/Device.h"

namespace arrus {

/**
 * An abstract base class for ultrasound devices.
 *
 * An ultrasound device allows performing sequence of Tx/Rx operations:
 * that is, generates ultrasound excitation and records the echo data.
 */
class UltrasoundDevice : Device {

    virtual void setTxRxSequence() = 0;
};

}

#endif //ARRUS_CORE_API_DEVICES_ULTRASOUNDDEVICE_H
