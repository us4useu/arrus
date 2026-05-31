#ifndef ARRUS_CORE_API_DEVICES_GPUSETTINGS_H
#define ARRUS_CORE_API_DEVICES_GPUSETTINGS_H

#include <optional>
#include <string>

#include "std4us/Optional.hpp"
#include "std4us/StdInterop.hpp"

namespace arrus::devices {

class GpuSettings {
public:
    explicit GpuSettings(std::optional<float> memoryLimitPercentage = std::nullopt, bool useMemoryPool = true)
        : memoryLimitPercentage(std4us::fromStd(memoryLimitPercentage)), useMemoryPool(useMemoryPool) {}

    // std4us-native constructor omitted: it collides with the std::optional
    // overload above when the first argument is implicitly convertible from
    // a bare float (e.g. proto-generated getters). Construct via std::optional
    // and let the inline shim convert.

    /**
     * Returns the GPU memory limit as a percentage of total VRAM (e.g., 0.5 = 50%).
     *
     * @return Memory limit percentage, or std::nullopt if no limit is set
     */
    std::optional<float> getMemoryLimitPercentage() const { return std4us::toStd(memoryLimitPercentage); }

    const std4us::Optional<float> &getMemoryLimitPercentageNative() const { return memoryLimitPercentage; }

    /**
     * Returns whether to use GPU memory pool.
     *
     * @return True if memory pool should be used, false otherwise
     */
    bool isUseMemoryPool() const { return useMemoryPool; }

    /**
     * Creates default GPU settings (no memory limit, memory pool enabled).
     *
     * @return Default GPU settings
     */
    static GpuSettings defaultSettings() {
        return GpuSettings();
    }

private:
    std4us::Optional<float> memoryLimitPercentage;
    bool useMemoryPool;
};

}// namespace arrus::devices

#endif//ARRUS_CORE_API_DEVICES_GPUSETTINGS_H
