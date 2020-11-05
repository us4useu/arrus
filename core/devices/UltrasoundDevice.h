#ifndef ARRUS_CORE_DEVICES_ULTRASOUNDDEVICE_H
#define ARRUS_CORE_DEVICES_ULTRASOUNDDEVICE_H

#include <vector>

#include "arrus/core/api/common/types.h"
#include "arrus/core/api/common/Interval.h"
#include "arrus/core/devices/TxRxParameters.h"
#include "arrus/core/api/ops/us4r/tgc.h"
#include "arrus/core/api/devices/us4r/FrameChannelMapping.h"
#include "arrus/core/devices/us4r/DataTransfer.h"

namespace arrus::devices {

class UltrasoundDevice {
public:
    virtual ~UltrasoundDevice() = default;

    /**
     * @param seq
     * @param tgcSamples
     * @param nRepeats how many times repeat the sequence of tx/rxs
     * @return
     */
     // TODO(pjarosik) wrap the below tuple in some meaning class
    virtual
    std::tuple<
        FrameChannelMapping::Handle,
        std::vector<std::vector<DataTransfer>>,
        uint16_t // ntriggers
    >
    setTxRxSequence(const std::vector<TxRxParameters> &seq,
                    const ::arrus::ops::us4r::TGCCurve &tgcSamples,
                    uint16 nRepeats) = 0;

    virtual void start() = 0;
    virtual void stop() = 0;

    virtual Interval<Voltage> getAcceptedVoltageRange() = 0;

    virtual void syncTrigger() = 0;
};

}

#endif //ARRUS_CORE_DEVICES_ULTRASOUNDDEVICE_H
