#ifndef ARRUS_CORE_API_DEVICES_GPU_H
#define ARRUS_CORE_API_DEVICES_GPU_H

#include "arrus/core/api/devices/Device.h"
#include "arrus/core/api/devices/GpuSettings.h"

#include <memory>

namespace arrus::devices {
class Gpu : public Device {
public:
    using Handle = std::unique_ptr<Gpu>;
    using RawHandle = PtrHandle<Gpu>;

    using Device::getDeviceId;

    Gpu(const DeviceId &id, const GpuSettings &settings) : Device(id), settings(settings) {}

    virtual std::string getDescription() const {
        return "General Purpose GPU";
    }

    const GpuSettings &getSettings() const { return settings; }

private:
    GpuSettings settings;
};

}




#endif //ARRUS_CORE_API_DEVICES_GPU_H
