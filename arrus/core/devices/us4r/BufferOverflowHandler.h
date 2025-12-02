#ifndef ARRUS_CORE_DEVICES_US4R_BUFFEROVERFLOWHANDLER_H
#define ARRUS_CORE_DEVICES_US4R_BUFFEROVERFLOWHANDLER_H

#include <vector>
#include <mutex>

#include "arrus/core/common/logging.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMImpl.h"


namespace arrus::devices::us4r {

/**
 * The class intended for handling buffer overflows in us4OEM devices.
 */
class BufferOverflowHandler {
public:

    BufferOverflowHandler(const std::vector<IUs4OEM*> &ius4oems) {
        this->ius4oems = ius4oems;
        this->logger = getLoggerFactory()->getLogger();
        INIT_ARRUS_DEVICE_LOGGER(this->logger, "BufferOverflowHandler");
    }

    void syncReceive(const uint16_t startIndex, const uint16_t endIndex) {
        std::unique_lock<std::mutex> guard(stateMutex);
        if(pendingIndexReceive.has_value()) {
            const auto value = pendingIndexReceive.value();
            if(startIndex <= value && value <= endIndex) {
                syncReceiveInternal();
            }
            pendingIndexReceive = std::nullopt;
        }
    }

    void syncTransfer(const uint16_t startIndex, const uint16_t endIndex) {
        std::unique_lock<std::mutex> guard(stateMutex);
        std::cout << "SYNC TRANSFER!" << std::endl;
        std::cout << "IS TRANSFER PENDING INDEX?"  << pendingIndexTransfer.has_value() << std::endl;
        if(pendingIndexTransfer.has_value()) {
            const auto value = pendingIndexTransfer.value();
            std::cout << "PENDING INDEX "  << value << " start index: " << startIndex << " end index: " << endIndex << std::endl;
            if(startIndex <= value && value <= endIndex) {
                std::cout << "calling sync transfer internal!" << std::endl;
                syncTransferInternal();
            }
            pendingIndexTransfer = std::nullopt;
        }
    }

    /**
     * Clears buffer overflow entries in the us4OEM sequence table.
     */
    void clearEntriesReceive(const uint16_t startFiring, const uint16_t endFiring) {
        std::unique_lock<std::mutex> guard(stateMutex);
        for (int i = (int) ius4oems.size() - 1; i >= 0; --i) {
            ius4oems[i]->MarkEntriesAsReadyForReceive(startFiring, endFiring);
        }
    }

    void clearEntriesTransfer(const uint16_t startFiring, const uint16_t endFiring) {
        std::unique_lock<std::mutex> guard(stateMutex);
        for (int i = (int) ius4oems.size() - 1; i >= 0; --i) {
            ius4oems[i]->MarkEntriesAsReadyForTransfer(startFiring, endFiring);
        }
    }

    void onReceiveOverflow() {
        std::unique_lock<std::mutex> guard(stateMutex);
        const uint16_t currentIdx = getMasterOEM()->GetSequencerCurrentIndex();

        if(pendingIndexReceive.has_value()) {
            // In that case, for some reason we are overriding, we expect the below to never happen.
            // TODO shouldn't we stop the system in that situation?
            logger->log(LogSeverity::ERROR,
                        format("There is already some RX pending index: previous: {}, new: {}", pendingIndexReceive.value(), currentIdx));
        }
        // TODO consider double checking that the current index is exactly the same on all the OEMs.
        // Check if this is spurious IRQ, i.e. if the flag of the entry at the current index is actually clean.
        if(getMasterOEM()->IsEntryReadyForReceive(currentIdx)) {
            if(!pendingIndexReceive.has_value()) {
                // If so, that could mean that all the interrupts 0 arrived earlier than this buffer overflow (???)
                // We expect that only sequence table flags were cleaned, but no "SyncReceive" was called.
                syncReceiveInternal();
            }
        }
        pendingIndexReceive = currentIdx;
    }

    void onTransferOverflow() {
        std::unique_lock<std::mutex> guard(stateMutex);
        const uint16_t currentIdx = getMasterOEM()->GetSequencerCurrentIndex();

        if(pendingIndexTransfer.has_value()) {
            // TODO shouldn't we stop the system in that situation?
            logger->log(LogSeverity::ERROR,
                        format("There is already some transfer pending index: previous: {}, new: {}", pendingIndexTransfer.value(), currentIdx));
        }
        // TODO consider double checking that the current index is exactly the same on all the OEMs.
        // Check if this is spurious IRQ, i.e. if the flag of the entry at the current index is actually clean.
        if(getMasterOEM()->IsEntryReadyForTransfer(currentIdx)) {
            // If so, that could mean that all the interrupts 0 arrived earlier than this buffer overflow (???)
            // We expect that only sequence table flags were cleaned, but no "SyncReceive" was called.
            if(!pendingIndexTransfer.has_value()) {
                syncTransferInternal();
            }
        }
        pendingIndexTransfer = currentIdx;
    }

    IUs4OEM* getMasterOEM() {
        return ius4oems[0];
    }

private:

    void syncReceiveInternal() {
        for (int i = (int) ius4oems.size() - 1; i >= 0; --i) {
            ius4oems[i]->SyncReceive();
        }
    }

    void syncTransferInternal() {
        for (int i = (int) ius4oems.size() - 1; i >= 0; --i) {
            ius4oems[i]->SyncTransfer();
        }
    }

    std::mutex stateMutex;
    Logger::Handle logger;
    std::vector<IUs4OEM*> ius4oems;
    std::optional<uint16_t> pendingIndexReceive;
    std::optional<uint16_t> pendingIndexTransfer;
};

}

#endif//ARRUS_CORE_DEVICES_US4R_BUFFEROVERFLOWHANDLER_H
