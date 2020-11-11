#ifndef ARRUS_CORE_DEVICES_US4R_US4R_H
#define ARRUS_CORE_DEVICES_US4R_US4R_H

#include <memory>

#include "arrus/core/api/devices/Device.h"
#include "arrus/core/api/devices/DeviceWithComponents.h"
#include "arrus/core/api/devices/us4r/Us4OEM.h"
#include "arrus/core/api/devices/us4r/ProbeAdapter.h"
#include "arrus/core/api/devices/probe/Probe.h"
#include "arrus/core/api/ops/us4r/TxRxSequence.h"
#include "FrameChannelMapping.h"
#include "HostBuffer.h"


namespace arrus::devices {

/**
 * Us4R system: a group of Us4OEM modules and related components.
 */
class Us4R : public DeviceWithComponents {
public:
    using Handle = std::unique_ptr<Us4R>;

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

    // TODO(pjarosik) return unique_ptr<FrameChannelMapping>
    // Currently shared_ptr is necessary for python swig wrappers only
    // TODO rxBufferSize, hostBufferSize
    virtual std::pair<
        std::shared_ptr<arrus::devices::FrameChannelMapping>,
        std::shared_ptr<arrus::devices::HostBuffer>
    >
    uploadSync(const ::arrus::ops::us4r::TxRxSequence &seq) = 0;

    /**
     *
     * @param seq
     * @param rxBufferSize
     * @param hostBufferSize
     * @param frameRepetitionInterval 0 means to not extend sequence interval time
     * @return
     */
    virtual std::pair<
        std::shared_ptr<arrus::devices::FrameChannelMapping>,
        std::shared_ptr<arrus::devices::HostBuffer>
    >
    uploadAsync(const ::arrus::ops::us4r::TxRxSequence &seq,
                unsigned short rxBufferSize,
                unsigned short hostBufferSize,
                float frameRepetitionInterval) = 0;

    virtual void setVoltage(Voltage voltage) = 0;

    virtual void disableHV() = 0;

    virtual void start() = 0;
    virtual void stop() = 0;

    Us4R(Us4R const&) = delete;
    Us4R(Us4R const&&) = delete;
    void operator=(Us4R const&) = delete;
    void operator=(Us4R const&&) = delete;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_US4R_H
