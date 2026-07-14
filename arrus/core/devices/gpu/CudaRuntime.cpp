#include "CudaRuntime.h"

#include "arrus/core/common/logging.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace arrus::devices {

namespace {

#ifdef _WIN32
using LibHandle = HMODULE;
#else
using LibHandle = void *;
#endif

// Values match cuda_runtime_api.h (CUDA 10/11/12).
using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
constexpr int cudaDevAttrIntegrated = 18;
constexpr int cudaDevAttrUnifiedAddressing = 41;
constexpr unsigned int cudaHostAllocDefault = 0;

using GetDeviceCountFn = cudaError_t (*)(int *);
using DeviceGetAttributeFn = cudaError_t (*)(int *, int, int);
using MallocFn = cudaError_t (*)(void **, std::size_t);
using HostAllocFn = cudaError_t (*)(void **, std::size_t, unsigned int);
using FreeFn = cudaError_t (*)(void *);
using FreeHostFn = cudaError_t (*)(void *);

LibHandle openLibrary() {
#ifdef _WIN32
    const char *candidates[] = {"cudart64_13.dll", "cudart64_12.dll", "cudart64_11.dll",
                                "cudart64_10.dll", "cudart.dll", nullptr};
    for (size_t i = 0; candidates[i] != nullptr; ++i) {
        HMODULE h = ::LoadLibraryA(candidates[i]);
        if (h != nullptr) {
            return h;
        }
    }
    return nullptr;
#else
    const char *candidates[] = {"libcudart.so",    "libcudart.so.13", "libcudart.so.12",
                                "libcudart.so.11", "libcudart.so.10", nullptr};
    for (size_t i = 0; candidates[i] != nullptr; ++i) {
        void *h = ::dlopen(candidates[i], RTLD_NOW | RTLD_GLOBAL);
        if (h != nullptr) {
            return h;
        }
    }
    return nullptr;
#endif
}

void closeLibrary(LibHandle h) {
    if (h == nullptr) {
        return;
    }
#ifdef _WIN32
    ::FreeLibrary(h);
#else
    ::dlclose(h);
#endif
}

void *resolveSymbol(LibHandle h, const char *name) {
#ifdef _WIN32
    return reinterpret_cast<void *>(::GetProcAddress(h, name));
#else
    return ::dlsym(h, name);
#endif
}

} // namespace

struct CudaRuntime::Impl {
    LibHandle handle = nullptr;
    GetDeviceCountFn getDeviceCount = nullptr;
    DeviceGetAttributeFn deviceGetAttribute = nullptr;
    MallocFn cudaMalloc = nullptr;
    HostAllocFn cudaHostAlloc = nullptr;
    FreeFn cudaFree = nullptr;
    FreeHostFn cudaFreeHost = nullptr;
    bool available = false;
};

CudaRuntime &CudaRuntime::instance() {
    static CudaRuntime inst;
    return inst;
}

CudaRuntime::CudaRuntime() : pImpl(new Impl) {
    pImpl->handle = openLibrary();
    if (pImpl->handle == nullptr) {
        getDefaultLogger()->debug("CUDA Runtime library not found; P2P DMA disabled.");
        return;
    }
    pImpl->getDeviceCount = reinterpret_cast<GetDeviceCountFn>(resolveSymbol(pImpl->handle, "cudaGetDeviceCount"));
    pImpl->deviceGetAttribute =
        reinterpret_cast<DeviceGetAttributeFn>(resolveSymbol(pImpl->handle, "cudaDeviceGetAttribute"));
    pImpl->cudaMalloc = reinterpret_cast<MallocFn>(resolveSymbol(pImpl->handle, "cudaMalloc"));
    pImpl->cudaHostAlloc = reinterpret_cast<HostAllocFn>(resolveSymbol(pImpl->handle, "cudaHostAlloc"));
    pImpl->cudaFree = reinterpret_cast<FreeFn>(resolveSymbol(pImpl->handle, "cudaFree"));
    pImpl->cudaFreeHost = reinterpret_cast<FreeHostFn>(resolveSymbol(pImpl->handle, "cudaFreeHost"));

    pImpl->available = pImpl->getDeviceCount != nullptr && pImpl->deviceGetAttribute != nullptr
        && pImpl->cudaMalloc != nullptr && pImpl->cudaHostAlloc != nullptr && pImpl->cudaFree != nullptr
        && pImpl->cudaFreeHost != nullptr;

    if (pImpl->available) {
        getDefaultLogger()->debug("CUDA Runtime library loaded.");
    } else {
        getDefaultLogger()->warn("CUDA Runtime library opened but a required symbol could not be resolved.");
    }
}

CudaRuntime::~CudaRuntime() {
    closeLibrary(pImpl->handle);
    delete pImpl;
}

bool CudaRuntime::isAvailable() const { return pImpl->available; }

int CudaRuntime::getDeviceCount() {
    if (!pImpl->available) {
        return 0;
    }
    int count = 0;
    if (pImpl->getDeviceCount(&count) != cudaSuccess) {
        return 0;
    }
    return count;
}

bool CudaRuntime::getDeviceIntegrated(int device) {
    if (!pImpl->available) {
        return false;
    }
    int value = 0;
    if (pImpl->deviceGetAttribute(&value, cudaDevAttrIntegrated, device) != cudaSuccess) {
        return false;
    }
    return value != 0;
}

bool CudaRuntime::getDeviceUnifiedAddressing(int device) {
    if (!pImpl->available) {
        return false;
    }
    int value = 0;
    if (pImpl->deviceGetAttribute(&value, cudaDevAttrUnifiedAddressing, device) != cudaSuccess) {
        return false;
    }
    return value != 0;
}

void *CudaRuntime::malloc(std::size_t size) {
    if (!pImpl->available) {
        return nullptr;
    }
    void *ptr = nullptr;
    if (pImpl->cudaMalloc(&ptr, size) != cudaSuccess) {
        return nullptr;
    }
    return ptr;
}

void *CudaRuntime::hostAllocDefault(std::size_t size) {
    if (!pImpl->available) {
        return nullptr;
    }
    void *ptr = nullptr;
    if (pImpl->cudaHostAlloc(&ptr, size, cudaHostAllocDefault) != cudaSuccess) {
        return nullptr;
    }
    return ptr;
}

void CudaRuntime::free(void *ptr) {
    if (!pImpl->available || ptr == nullptr) {
        return;
    }
    pImpl->cudaFree(ptr);
}

void CudaRuntime::freeHost(void *ptr) {
    if (!pImpl->available || ptr == nullptr) {
        return;
    }
    pImpl->cudaFreeHost(ptr);
}

} // namespace arrus::devices
