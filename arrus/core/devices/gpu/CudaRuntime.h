#ifndef ARRUS_CORE_DEVICES_GPU_CUDARUNTIME_H
#define ARRUS_CORE_DEVICES_GPU_CUDARUNTIME_H

#include <cstddef>

namespace arrus::devices {

/**
 * Weak CUDA Runtime API.
 *
 * The CUDA Runtime library is opened with dlopen / LoadLibrary on first access.
 */
class CudaRuntime {
public:
    static CudaRuntime &instance();

    bool isAvailable() const;

    int getDeviceCount();
    bool getDeviceIntegrated(int device);
    bool getDeviceUnifiedAddressing(int device);

    void *malloc(std::size_t size);
    void *hostAllocDefault(std::size_t size);
    void free(void *ptr);
    void freeHost(void *ptr);

private:
    CudaRuntime();
    ~CudaRuntime();
    CudaRuntime(const CudaRuntime &) = delete;
    CudaRuntime &operator=(const CudaRuntime &) = delete;

    struct Impl;
    Impl *pImpl;
};

} // namespace arrus::devices

#endif // ARRUS_CORE_DEVICES_GPU_CUDARUNTIME_H
