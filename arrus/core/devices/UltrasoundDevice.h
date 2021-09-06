#ifndef ARRUS_CORE_DEVICES_ULTRASOUNDDEVICE_H
#define ARRUS_CORE_DEVICES_ULTRASOUNDDEVICE_H

#include <vector>

#include "arrus/core/api/common/types.h"
#include "arrus/core/api/common/Interval.h"
#include "arrus/core/devices/TxRxParameters.h"
#include "arrus/core/api/ops/us4r/tgc.h"
#include "arrus/core/api/devices/us4r/FrameChannelMapping.h"
#include "arrus/core/devices/us4r/DataTransfer.h"
#include "arrus/core/devices/us4r/Us4ROutputBuffer.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMBuffer.h"

namespace arrus::devices {

class UltrasoundDevice {
public:
    virtual ~UltrasoundDevice() = default;

    virtual void start() = 0;

    virtual void stop() = 0;

    virtual Interval<Voltage> getAcceptedVoltageRange() = 0;

    virtual void syncTrigger() = 0;
};

}

#endif //ARRUS_CORE_DEVICES_ULTRASOUNDDEVICE_H
