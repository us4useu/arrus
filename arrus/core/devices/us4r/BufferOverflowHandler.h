#ifndef ARRUS_CORE_DEVICES_US4R_BUFFEROVERFLOWHANDLER_H
#define ARRUS_CORE_DEVICES_US4R_BUFFEROVERFLOWHANDLER_H

namespace arrus::devices::us4r {

/**
 * The class intended for handling buffer overflows in us4OEM devices.
 */
class BufferOverflowHandler {
public:


    void sync() {
        if(pendingIndexReceive.has_value()) {
            const auto value = pending.value();
            if(startIndex >= value && endIndex <= value) {
                syncReceive();
            }
            pendingIndexReceive = std::nullopt;
        }

        if(pendingIndexTransfer.has_value()) {
            const auto value = pending.value();
            if(startIndex >= value && endIndex <= value) {
                syncTransfer();
            }
            pendingIndexTransfer = std::nullopt;
        }
    }

    /**
     * Clears buffer overflow entries in the us4OEM sequence table.
     */
    void clearEntries(const uint16_t startIndex, const uint16_t endIndex) {
        std::unique_lock<std::mutex> guard(stateMutex);
        clearReceiveAndTransfer(startIndex, endIndex);
    }

    void onReceiveOverflow() {
        std::unique_lock<std::mutex> guard(stateMutex);
        if(isEntryClean()) {
            // we already have the entry clean (spurious IRQ?)
            notify
        }
        registerReceiveOverflowEvent();
    }

private:
    void registerReceiveOverflowEvent(us4oem) {
        const auto currentIndex = getCurrentIndex();
        if(pendingIndexReceive.has_value()) {
            logger->log(
                ERROR,
                format("There is already some RX pending index: previous: {}, new: {}", pendingIndexReceive.value(), currentIndex)
            );
        }
        pendingIndexReceive = currentIndex;
    }

    void registerTransferOverflowEvent() {
        std::unique_lock<std::mutex> guard(stateMutex);

        const auto currentIndex = getCurrentIndex();
        if(pendingIndexTransfer.has_value()) {
            logger->log(
                ERROR,
                format("There is already some PCIEDMA pending index: previous: {}, new: {}", pendingIndexTransfer.value(), currentIndex)
            );
        }
        pendingIndexTransfer = currentIndex;
    }

    std::mutex stateMutex;
    std::vector<IUs4OEM*> ius4oems;
    std::function<uint16_t()> getCurrentIndex;
};

}

#endif//ARRUS_CORE_DEVICES_US4R_BUFFEROVERFLOWHANDLER_H
