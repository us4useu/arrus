#ifndef ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMIMPLBASE_H
#define ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMIMPLBASE_H

#include <vector>
#include "arrus/core/devices/us4r/external/ius4oem/IUs4OEMFactory.h"
#include "arrus/core/api/devices/us4r/RxSettings.h"
#include "arrus/core/api/devices/us4r/FrameChannelMapping.h"
#include "arrus/core/api/devices/us4r/Us4OEM.h"
#include "arrus/core/devices/TxRxParameters.h"
#include "arrus/core/api/ops/us4r/tgc.h"
#include "arrus/core/devices/UltrasoundDevice.h"

namespace arrus::devices {

class Us4OEMImplBase : public Us4OEM, public UltrasoundDevice {
public:
    using Handle = std::unique_ptr<Us4OEMImplBase>;
    using RawHandle = PtrHandle<Us4OEMImplBase>;

    ~Us4OEMImplBase() override = default;

    Us4OEMImplBase(Us4OEMImplBase const &) = delete;

    Us4OEMImplBase(Us4OEMImplBase const &&) = delete;

    void operator=(Us4OEMImplBase const &) = delete;

    void operator=(Us4OEMImplBase const &&) = delete;

    void syncTrigger() override = 0;

    virtual bool isMaster() = 0;

    virtual std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle>
    setTxRxSequence(const std::vector<TxRxParameters> &seq, const ops::us4r::TGCCurve &tgcSamples, uint16 rxBufferSize,
                    uint16 rxBatchSize, std::optional<float> sri, bool triggerSync,
                    const std::optional<::arrus::ops::us4r::DigitalDownConversion> &ddc,
                    const std::vector<arrus::framework::NdArray> &txDelays) = 0;

    // TODO expose "registerUs4OEMOutputBuffer" function, keep this class hermetic
    virtual Ius4OEMRawHandle getIUs4oem() = 0;

    virtual void enableSequencer(bool resetSequencerPointer = true) = 0;

    virtual std::vector<uint8_t> getChannelMapping() = 0;

    virtual void setRxSettings(const RxSettings& settings) = 0;

    virtual void setTestPattern(RxTestPattern pattern) = 0;

    virtual void setSubsequence(uint16 start, uint16 end, bool syncMode, const std::optional<float> &sri) = 0;

    virtual void clearCallbacks() = 0;

protected:
    explicit Us4OEMImplBase(const DeviceId &id) : Us4OEM(id) {}
};

}

#endif //ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMIMPLBASE_H
