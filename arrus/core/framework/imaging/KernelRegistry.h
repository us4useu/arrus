#ifndef CPP_EXAMPLE_IMAGING_KERNELREGISTRY_H
#define CPP_EXAMPLE_IMAGING_KERNELREGISTRY_H

#include <functional>
#include <iostream>

#include "imaging/Kernel.h"
#include "imaging/Operation.h"

namespace arrus_example_imaging {
class KernelRegistry {
public:
    using KernelFactory = std::function<Kernel::Handle(KernelConstructionContext &)>;

    static KernelRegistry &getInstance() {
        static KernelRegistry instance;
        return instance;
    }

    Kernel::Handle createKernel(const Operation &op, KernelConstructionContext &ctx) {
        KernelFactory func;
        try {
            func = kernels.at(op.getOpClassId());
        } catch (const std::out_of_range &e) {
            throw std::out_of_range("There is no kernel registered for operation: " + op.getOpClassId());
        }
        return func(ctx);
    }

    void registerKernelOpFactory(const Operation::OpClassId &classId, KernelFactory factory) {
        // TODO wrap it as a macro, that can be called in .cpp
        // TODO check if a kernel with given class id already exists (if so, throw excception)
        kernels.insert({classId, factory});
    }

private:
    // Factory functions.
    std::unordered_map<Operation::OpClassId, KernelFactory> kernels;
};

template<typename T> class RegisterKernelOpInitializer {
public:
    explicit RegisterKernelOpInitializer(const Operation::OpClassId id) {
        try {
            KernelRegistry::getInstance().registerKernelOpFactory(
                id, [](KernelConstructionContext &ctx) { return std::make_unique<T>(ctx); });
        } catch(const std::exception &e) {
            std::cerr << "Error while registering op kernel: " << e.what() << std::endl;
        } catch(...) {
            std::cerr << "Unknown error while registering op kernel" << std::endl;
        }
    }
};

#define REGISTER_KERNEL_OP(opClassId, KernelClass) \
namespace {                                        \
    static RegisterKernelOpInitializer<KernelClass> opInitializer{opClassId}; \
}

}// namespace arrus_example_imaging

#endif//CPP_EXAMPLE_IMAGING_KERNELREGISTRY_H
