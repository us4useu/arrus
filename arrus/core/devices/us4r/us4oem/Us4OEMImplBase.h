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
#include "Us4OEMUploadResult.h"

namespace arrus::devices {

class Us4OEMImplBase : public Us4OEM, public UltrasoundDevice {
public:
    using Handle = std::unique_ptr<Us4OEMImplBase>;
    using RawHandle = PtrHandle<Us4OEMImplBase>;

    static constexpr ChannelIdx N_TX_CHANNELS = 128;
    static constexpr ChannelIdx N_RX_CHANNELS = 32;
    static constexpr ChannelIdx N_ADDR_CHANNELS = N_TX_CHANNELS;

    ~Us4OEMImplBase() override = default;

    Us4OEMImplBase(Us4OEMImplBase const &) = delete;

    Us4OEMImplBase(Us4OEMImplBase const &&) = delete;

    void operator=(Us4OEMImplBase const &) = delete;

    void operator=(Us4OEMImplBase const &&) = delete;

    void syncTrigger() override = 0;

    virtual bool isMaster() = 0;

    virtual Us4OEMUploadResult
    upload(const std::vector<us4r::TxRxParametersSequence> &sequences,
           uint16 rxBufferSize, ops::us4r::Scheme::WorkMode workMode,
           const std::optional<ops::us4r::DigitalDownConversion> &ddc,
           const std::vector<arrus::framework::NdArray> &txDelays) = 0;
    virtual Ius4OEMRawHandle getIUs4OEM() = 0;
    virtual void enableSequencer(bool resetSequencerPointer) = 0;
    virtual std::vector<uint8_t> getChannelMapping() = 0;
    virtual void setRxSettings(const RxSettings& settings) = 0;
    virtual void setTestPattern(RxTestPattern pattern) = 0;
    virtual BitstreamId addIOBitstream(const std::vector<uint8_t> &levels, const std::vector<uint16_t> &periods) = 0;
    virtual void setIOBitstream(BitstreamId id, const std::vector<uint8_t> &levels, const std::vector<uint16_t> &periods) = 0;
    virtual void clearCallbacks() = 0;

protected:
    explicit Us4OEMImplBase(const DeviceId &id) : Us4OEM(id) {}
};

}

#endif //ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMIMPLBASE_H
