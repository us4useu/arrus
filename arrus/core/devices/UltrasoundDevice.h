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

namespace arrus::devices {

class UltrasoundDevice {
public:
    virtual ~UltrasoundDevice() = default;

     // TODO(pjarosik) wrap the below tuple in some meaningful class
    virtual
     std::tuple<FrameChannelMapping::Handle, std::vector<std::vector<DataTransfer>>>
    setTxRxSequence(const std::vector<TxRxParameters> &seq, const ops::us4r::TGCCurve &tgcSamples,
                    uint16 rxBufferSize, uint16 rxBatchSize, std::optional<float> sri) = 0;

    virtual void start() = 0;

    virtual void stop() = 0;

    virtual Interval<Voltage> getAcceptedVoltageRange() = 0;

    virtual void syncTrigger() = 0;

    virtual void registerOutputBuffer(Us4ROutputBuffer* outputBuffer,
                                      const std::vector<std::vector<DataTransfer>> &transfers) = 0;

    virtual void setTgcCurve(const ::arrus::ops::us4r::TGCCurve &tgcCurve) = 0;
};

}

#endif //ARRUS_CORE_DEVICES_ULTRASOUNDDEVICE_H
