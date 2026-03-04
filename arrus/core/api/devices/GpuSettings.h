#ifndef ARRUS_CORE_API_DEVICES_GPUSETTINGS_H
#define ARRUS_CORE_API_DEVICES_GPUSETTINGS_H

#include <string>
#include <optional>

namespace arrus::devices {

class GpuSettings {
public:
    explicit GpuSettings(std::optional<float> memoryLimitPercentage = std::nullopt, bool useMemoryPool = true, bool useP2pDma = false)
        : memoryLimitPercentage(std::move(memoryLimitPercentage)), useMemoryPool(useMemoryPool), useP2pDma(useP2pDma) {}

    /**
     * Returns the GPU memory limit as a percentage of total VRAM (e.g., 0.5 = 50%).
     * 
     * @return Memory limit percentage, or std::nullopt if no limit is set
     */
    const std::optional<float>& getMemoryLimitPercentage() const { return memoryLimitPercentage; }

    /**
     * Returns whether to use P2P DMA.
     * 
     * @return True if P2P DMA should be used, false otherwise
     */
    bool usesP2pDma() const { return useP2pDma; }

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
    bool useP2pDma;
};

}

#endif //ARRUS_CORE_API_DEVICES_GPUSETTINGS_H