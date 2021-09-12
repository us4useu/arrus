#ifndef ARRUS_CORE_DEVICES_US4R_DATATRANSFER_H
#define ARRUS_CORE_DEVICES_US4R_DATATRANSFER_H

#include <functional>

namespace arrus::devices {

class DataTransfer {
public:
    DataTransfer(std::function<void(uint8_t *)> transferFunc, size_t size, size_t srcAddress, uint16 firing)
        : transferFunc(std::move(transferFunc)), size(size), srcAddress(srcAddress), firing(firing) {}

    [[nodiscard]] const std::function<void(uint8_t *)> &getTransferFunc() const {
        return transferFunc;
    }

    [[nodiscard]] size_t getSize() const {
        return size;
    }

    [[nodiscard]] size_t getSrcAddress() const {
        return srcAddress;
    }

    [[nodiscard]] uint16 getFiring() const {
        return firing;
    }

private:
    std::function<void(uint8_t *)> transferFunc;
    size_t size;
    size_t srcAddress;
    uint16 firing;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_DATATRANSFER_H
