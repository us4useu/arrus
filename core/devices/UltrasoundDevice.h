#ifndef ARRUS_CORE_DEVICES_ULTRASOUNDDEVICE_H
#define ARRUS_CORE_DEVICES_ULTRASOUNDDEVICE_H

#include <vector>

#include "arrus/core/api/common/types.h"
#include "arrus/core/api/common/Interval.h"
#include "arrus/core/devices/TxRxParameters.h"
#include "arrus/core/api/ops/us4r/tgc.h"

namespace arrus::devices {

class UltrasoundDevice {
public:
    virtual ~UltrasoundDevice() = default;

    virtual void
    setTxRxSequence(const std::vector<TxRxParameters> &seq,
                    const ::arrus::ops::us4r::TGCCurve &tgcSamples) = 0;

    virtual Interval<Voltage> getAcceptedVoltageRange() = 0;
};

}

#endif //ARRUS_CORE_DEVICES_ULTRASOUNDDEVICE_H
