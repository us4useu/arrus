#ifndef ARRUS_CORE_API_DEVICES_ULTRASOUND_H
#define ARRUS_CORE_API_DEVICES_ULTRASOUND_H

#include <memory>


#include "arrus/core/api/devices/Device.h"
#include "arrus/core/api/devices/DeviceWithComponents.h"
#include "arrus/core/api/devices/probe/Probe.h"
#include "arrus/core/api/devices/us4r/Us4OEM.h"
#include "arrus/core/api/framework/Buffer.h"
#include "arrus/core/api/framework/DataBufferSpec.h"
// TODO(pjarosik) avoid using us4r specific objects here
#include "arrus/core/api/ops/us4r/Scheme.h"
#include "arrus/core/api/ops/us4r/TxRxSequence.h"
#include "arrus/core/api/devices/us4r/FrameChannelMapping.h"
#include "arrus/core/api/devices/us4r/RxSettings.h"

namespace arrus::devices {

/**
 * An interface to the ultrasound device.
 */
class Ultrasound: public Device {
public:
    using Handle = std::unique_ptr<Ultrasound>;

    explicit Ultrasound(const DeviceId &id) : Device(id) {}

    virtual ~Ultrasound() = default;

    virtual std::pair<
        std::shared_ptr<arrus::framework::Buffer>,
        std::shared_ptr<arrus::session::Metadata>
    >
    upload(const ::arrus::ops::us4r::Scheme &scheme) = 0;

    virtual void start() = 0;
    virtual void stop() = 0;

    /**
     * Returns NOMINAL Ultrasound device sampling frequency.
     */
    virtual float getSamplingFrequency() const = 0;

    Ultrasound(Ultrasound const &) = delete;
    Ultrasound(Ultrasound const &&) = delete;
    void operator=(Ultrasound const &) = delete;
    void operator=(Ultrasound const &&) = delete;
};

}// namespace arrus::devices

#endif//ARRUS_CORE_API_DEVICES_ULTRASOUND_H
