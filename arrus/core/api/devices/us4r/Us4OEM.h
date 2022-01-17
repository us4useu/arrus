#ifndef ARRUS_CORE_API_DEVICES_US4R_US4OEM_H
#define ARRUS_CORE_API_DEVICES_US4R_US4OEM_H

#include <memory>
#include "arrus/core/api/devices/Device.h"
#include "arrus/core/api/common/types.h"
#include "arrus/core/api/devices/TriggerGenerator.h"

namespace arrus::devices {

class Us4OEM : public Device, public TriggerGenerator {
public:
    using Handle = std::unique_ptr<Us4OEM>;
    using RawHandle = PtrHandle<Us4OEM>;

    ~Us4OEM() override = default;

    virtual double getSamplingFrequency() = 0;

    virtual float getFPGATemperature() = 0;

	virtual uint16_t getAfe(uint8_t address) = 0;

	virtual void setAfe(uint8_t address, uint16_t value) = 0;

    Us4OEM(Us4OEM const&) = delete;
    Us4OEM(Us4OEM const&&) = delete;
    void operator=(Us4OEM const&) = delete;
    void operator=(Us4OEM const&&) = delete;
protected:
    explicit Us4OEM(const DeviceId &id): Device(id) {}
};


}

#endif // ARRUS_CORE_API_DEVICES_US4R_US4OEM_H
