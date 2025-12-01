#ifndef ARRUS_CORE_DEVICES_US4R_BUFFEROVERFLOWHANDLER_H
#define ARRUS_CORE_DEVICES_US4R_BUFFEROVERFLOWHANDLER_H

namespace arrus::devices::us4r {

/**
 * The class intended for handling buffer overflows in us4OEM devices.
 */
class BufferOverflowHandler {
public:

    void syncReceive(const uint16_t startIndex, const uint16_t endIndex) {
        if(pendingIndexReceive.has_value()) {
            const auto value = pendingIndexReceive.value();
            if(startIndex >= value && value >= endIndex) {
                syncReceiveInternal();
            }
            pendingIndexReceive = std::nullopt;
        }
    }

    void syncTransfer(const uint16_t startIndex, const uint16_t endIndex) {
        if(pendingIndexTransfer.has_value()) {
            const auto value = pendingIndexTransfer.value();
            if(startIndex >= value && value >= endIndex) {
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
        // TODO consider double checking that the current index is exactly the same on all the OEMs.
        // Check if this is spurious IRQ, i.e. if the flag of the entry at the current index is actually clean.
        if(getMasterOEM()->IsEntryReadyForReceive(currentIdx)) {
            return;
        }
        if(pendingIndexReceive.has_value()) {
            // TODO shouldn't we stop the system in that situation?
            logger->log(LogSeverity::ERROR,
                        format("There is already some RX pending index: previous: {}, new: {}", pendingIndexReceive.value(), currentIdx));
        }
        pendingIndexReceive = currentIdx;
    }

    void onTransferOverflow() {
        std::unique_lock<std::mutex> guard(stateMutex);
        const uint16_t currentIdx = getMasterOEM()->GetSequencerCurrentIndex();
        // TODO consider double checking that the current index is exactly the same on all the OEMs.
        // Check if this is spurious IRQ, i.e. if the flag of the entry at the current index is actually clean.
        if(getMasterOEM()->IsEntryReadyForTransfer(currentIdx)) {
            // The entry is already ready for transfer, nothing to do more on that.
            return;
        }
        if(pendingIndexTransfer.has_value()) {
            // TODO shouldn't we stop the system in that situation?
            logger->log(LogSeverity::ERROR,
                        format("There is already some Transfer pending index: previous: {}, new: {}", pendingIndexTransfer.value(), currentIdx));
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
