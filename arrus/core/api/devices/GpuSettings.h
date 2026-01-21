#ifndef ARRUS_CORE_API_DEVICES_GPUSETTINGS_H
#define ARRUS_CORE_API_DEVICES_GPUSETTINGS_H

#include <string>
#include <optional>

namespace arrus::devices {

class GpuSettings {
public:
    explicit GpuSettings(std::optional<float> memoryLimitPercentage = std::nullopt, bool useMemoryPool = true)
        : memoryLimitPercentage(std::move(memoryLimitPercentage)), useMemoryPool(useMemoryPool) {}

    /**
     * Returns the GPU memory limit as a percentage of total VRAM (e.g., 0.5 = 50%).
     * 
     * @return Memory limit percentage, or std::nullopt if no limit is set
     */
    const std::optional<float>& getMemoryLimitPercentage() const { return memoryLimitPercentage; }


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
    std::optional<float> memoryLimitPercentage;
    bool useMemoryPool;
};

}

#endif //ARRUS_CORE_API_DEVICES_GPUSETTINGS_H