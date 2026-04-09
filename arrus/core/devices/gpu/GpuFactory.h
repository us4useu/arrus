#ifndef ARRUS_CORE_DEVICES_GPU_GPUFACTORY_H
#define ARRUS_CORE_DEVICES_GPU_GPUFACTORY_H

#include <memory>

#include "arrus/core/api/devices/Gpu.h"
#include "arrus/core/api/devices/GpuSettings.h"

namespace arrus::devices {
class GpuFactory {
public:
    using Handle = std::unique_ptr<GpuFactory>;

    virtual Gpu::Handle getGpu(Ordinal ordinal, const GpuSettings &settings) {
        return std::make_unique<Gpu>(DeviceId(DeviceType::GPU, ordinal), settings);
    }

    virtual ~GpuFactory() = default;
};

}

#endif//ARRUS_CORE_DEVICES_GPU_GPUFACTORY_H
