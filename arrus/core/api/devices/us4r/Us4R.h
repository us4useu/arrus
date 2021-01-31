#ifndef ARRUS_CORE_DEVICES_US4R_US4R_H
#define ARRUS_CORE_DEVICES_US4R_US4R_H

#include <memory>

#include "arrus/core/api/devices/Device.h"
#include "arrus/core/api/devices/DeviceWithComponents.h"
#include "arrus/core/api/devices/us4r/Us4OEM.h"
#include "arrus/core/api/devices/us4r/ProbeAdapter.h"
#include "arrus/core/api/devices/probe/Probe.h"
#include "arrus/core/api/ops/us4r/TxRxSequence.h"
#include "arrus/core/api/ops/us4r/Scheme.h"
#include "arrus/core/api/framework/DataBuffer.h"
#include "arrus/core/api/framework/DataBufferSpec.h"
#include "FrameChannelMapping.h"


namespace arrus::devices {

/**
 * Us4R system: a group of Us4OEM modules and related components.
 */
class Us4R : public DeviceWithComponents {
public:
    using Handle = std::unique_ptr<Us4R>;
    static constexpr long long INF_TIMEOUT = -1;

    explicit Us4R(const DeviceId &id): DeviceWithComponents(id) {}

    ~Us4R() override = default;

    /**
     * Returns a handle to Us4OEM identified by given ordinal number.
     *
     * @param ordinal ordinal number of the us4oem to get
     * @return a handle to the us4oem module
     */
    virtual Us4OEM::RawHandle getUs4OEM(Ordinal ordinal) = 0;

    /**
     * Returns a handle to an adapter identified by given ordinal number.
     *
     * @param ordinal ordinal number of the adapter to get
     * @return a handle to the adapter device
     */
    virtual ProbeAdapter::RawHandle getProbeAdapter(Ordinal ordinal) = 0;

    /**
     * Returns a handle to a probe identified by given ordinal number.
     *
     * @param ordinal ordinal number of the probe to get
     * @return a handle to the probe
     */
    virtual arrus::devices::Probe* getProbe(Ordinal ordinal) = 0;

    /**
     *  Uploads a given
     *
     * @param seq
     * @param rxBufferSize
     * @param hostBufferSize
     * @return
     */
    virtual std::pair<
        std::shared_ptr<arrus::framework::DataBuffer>,
        std::shared_ptr<arrus::devices::FrameChannelMapping>
    >
    upload(const ::arrus::ops::us4r::TxRxSequence &seq, unsigned short rxBufferSize,
           const ::arrus::ops::us4r::Scheme::WorkMode &workMode,
           const ::arrus::framework::DataBufferSpec &hostBufferSpec) = 0;

    virtual void setVoltage(Voltage voltage) = 0;

    virtual void disableHV() = 0;

    /**
     * Sets tgc curve points asynchronously.
     *
     * @param tgcCurvePoints tgc curve points to set.
     */
    virtual void setTgcCurve(const std::vector<float>& tgcCurvePoints) = 0;

    virtual void start() = 0;
    virtual void stop() = 0;

    Us4R(Us4R const&) = delete;
    Us4R(Us4R const&&) = delete;
    void operator=(Us4R const&) = delete;
    void operator=(Us4R const&&) = delete;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_US4R_H
