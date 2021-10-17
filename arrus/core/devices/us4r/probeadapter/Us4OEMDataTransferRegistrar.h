#ifndef ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_US4OEMDATATRANSFERREGISTRAR_H
#define ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_US4OEMDATATRANSFERREGISTRAR_H

#include "arrus/core/devices/us4r/us4oem/Us4OEMImplBase.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMImpl.h"
#include "arrus/core/devices/us4r/Us4ROutputBuffer.h"
#include "arrus/core/api/common/types.h"
#include "arrus/core/common/logging.h"

namespace arrus::devices {

class Transfer {
public:
    Transfer(size_t address, size_t size, uint16 firing) : address(address), size(size), firing(firing) {}

    bool operator==(const Transfer &rhs) const {
        return address == rhs.address && size == rhs.size && firing == rhs.firing;
    }
    bool operator!=(const Transfer &rhs) const {
        return !(rhs == *this);
    }

    size_t address{0};
    size_t size{0};
    uint16 firing{0};
};

/**
 * Registers transfers from us4R internal DDR memory to the destination (host) memory.
 */
class Us4OEMDataTransferRegistrar {
public:
    static constexpr size_t MAX_N_TRANSFERS = 256;
    static constexpr size_t MAX_TRANSFER_SIZE = Us4OEMImpl::MAX_TRANSFER_SIZE;

    Us4OEMDataTransferRegistrar(Us4ROutputBuffer *dst, const Us4OEMBuffer *src, Us4OEMImplBase *us4oem)
            : logger(loggerFactory->getLogger()), dstBuffer(dst), srcBuffer(src), us4oem(us4oem) {
        ARRUS_INIT_COMPONENT_LOGGER(logger, "Us4OEMDataTransferRegistrar");
        if (dst->getNumberOfElements() % src->getNumberOfElements() != 0) {
            throw IllegalArgumentException("Host buffer should have multiple of rx buffer elements.");
        }
        ius4oem = us4oem->getIUs4oem();
        us4oemOrdinal = us4oem->getDeviceId().getOrdinal();
        elementTransfers = groupPartsIntoTransfers(src->getElementParts());

        srcNElements = src->getNumberOfElements();
        dstNElements = dst->getNumberOfElements();
        nTransfersPerElement = elementTransfers.size();
        // Number of transfer src points.
        srcNTransfers = nTransfersPerElement*srcNElements; // Should be <= 256
        // Number of transfer dst points.
        dstNTransfers = nTransfersPerElement*dstNElements; // Can be > 256

        ARRUS_REQUIRES_AT_MOST(srcNTransfers, MAX_N_TRANSFERS, "Exceeded maximum number of transfers.");

        // If true: create only nSrc transfers, the callback function will reprogram the appropriate number transfers.
        reprogramTransfers = dstNTransfers > MAX_N_TRANSFERS;

        if(dstNTransfers > MAX_N_TRANSFERS) {
            reprogramTransfers = true;
            strategy = 2;
        }
        else if(dstNTransfers > srcNTransfers) {
            // reschedule needed
            strategy = 1;
        }
        else {
            // nTransferDst == nTransferSrc
            strategy = 0;
        }
    }

    void registerTransfers() {
        // Page-lock all host dest points.
        pageLockDstMemory();

        // Send page descriptors to us4OEM DMA.
        size_t nSrcPoints = srcNTransfers;
        size_t nDstPoints = reprogramTransfers ? srcNTransfers : dstNTransfers;
        programTransfers(nSrcPoints, nDstPoints);
        scheduleTransfers();
    }

    static std::vector<Transfer> groupPartsIntoTransfers(const std::vector<Us4OEMBufferElementPart> &parts) {
        std::vector<Transfer> transfers;
        size_t address = 0;
        size_t size = 0;
        size_t firing = 0; // the firing that finishes given transfer
        for(auto &part: parts) {
            // Assumption: size of each part is less than the possible maximum
            if(size + part.getSize() > MAX_TRANSFER_SIZE) {
                transfers.emplace_back(address, size, firing);
                address = part.getAddress();
                size = 0;
            }
            size += part.getSize();
            firing = part.getFiring();
        }
        if(size > 0) {
            transfers.emplace_back(address, size, firing);
        }
        return transfers;
    }

    void pageLockDstMemory() {
        for(size_t dstIdx = 0, srcIdx = 0; dstIdx < dstNElements; ++dstIdx, srcIdx = (srcIdx+1) % srcNElements) {
            uint8 *addressDst = dstBuffer->getAddress(dstIdx, us4oemOrdinal);
            size_t addressSrc = srcBuffer->getElement(srcIdx).getAddress(); // bytes addressed
            for(auto &transfer: elementTransfers) {
                uint8 *dst = addressDst + transfer.address;
                size_t src = addressSrc + transfer.address;
                size_t size = transfer.size;
                ius4oem->PrepareHostBuffer(dst, size, src);
            }
        }
    }

    void programTransfers(size_t nSrcPoints, size_t nDstPoints) {
        for(size_t dstIdx = 0, srcIdx = 0; dstIdx < nDstPoints; ++dstIdx, srcIdx = (srcIdx+1) % nSrcPoints) {
            uint8 *addressDst = dstBuffer->getAddress(dstIdx, us4oemOrdinal);
            size_t addressSrc = srcBuffer->getElement(srcIdx).getAddress(); // bytes addressed
            for(size_t localTransferIdx = 0; localTransferIdx < nTransfersPerElement; ++localTransferIdx) {
                auto &transfer = elementTransfers[localTransferIdx];
                size_t transferIdx = srcIdx * nTransfersPerElement + localTransferIdx; // global transfer idx
                uint8 *dst = addressDst + transfer.address;
                size_t src = addressSrc + transfer.address;
                size_t size = transfer.size;
                ius4oem->PrepareTransferRXBufferToHost(transferIdx, dst, size, src);
            }
        }
    }

// ON NEW DATA CALLBACK POLICIES
// TODO replace macros with templates after refactoring us4r-api
#define ON_NEW_DATA_CALLBACK_signal_true \
    dstBuffer->signal(us4oemOrdinal, currentDstIdx); \
    currentDstIdx = (currentDstIdx + srcNElements) % dstNElements;

#define ON_NEW_DATA_CALLBACK_signal_false

// Strategy 0: keep transfers as they are (nSrc == nDst)
#define ON_NEW_DATA_CALLBACK_strategy_0

// Strategy 1: change sequencer firings definition, so the next firing will trigger the next portion of transfers
// (nSrc < nDst && nDst <= 256)
#define ON_NEW_DATA_CALLBACK_strategy_1 \
    currentTransferIdx = (currentTransferIdx + srcNTransfers) % dstNTransfers; \
    ius4oem->ScheduleTransferRXBufferToHost(transferLastFiring, currentTransferIdx, nullptr);

// Strategy 2: change transfer definition, so in the next call this transfer will write to subsequent dst element
// (nDst > 256)
#define ON_NEW_DATA_CALLBACK_strategy_2 \
    uint16 nextElementIdx = (currentDstIdx + srcNElements) % dstNElements; \
    auto nextDstAddress = dstBuffer->getAddress(nextElementIdx, us4oemOrdinal); \
    nextDstAddress += transfer.address;                                    \
    ius4oem->PrepareTransferRXBufferToHost(currentTransferIdx, nextDstAddress, transferSize, src);


#define ON_NEW_DATA_CALLBACK(signal, strategy) \
[=, currentDstIdx = srcIdx, currentTransferIdx = transferIdx] () mutable { \
    try {                            \
        ius4oem->MarkEntriesAsReadyForReceive(transferFirstFiring, transferLastFiring); \
        ON_NEW_DATA_CALLBACK_strategy_##strategy                             \
        ON_NEW_DATA_CALLBACK_signal_##signal                       \
    } \
    catch (const std::exception &e) { \
        logger->log(LogSeverity::ERROR, format("Us4OEM {}: callback exception: {}", us4oemOrdinal, e.what())); \
    } catch (...) { \
        logger->log(LogSeverity::ERROR, format("Us4OEM {}: callback unknown exception.", us4oemOrdinal)); \
    } \
}
    void scheduleTransfers() {
        // Here schedule transfers only from the start points (nSrc calls), dst pointers will be incremented
        // appropriately (if necessary).
        uint16 elementFirstFiring = 0;
        for(size_t srcIdx = 0; srcIdx < srcNTransfers; ++srcIdx) {
            size_t addressSrc = srcBuffer->getElement(srcIdx).getAddress(); // bytes addressed
            uint16 elementLastFiring = srcBuffer->getElement(srcIdx).getFiring();
            // for each element's part transfer:
            uint16 transferFirstFiring = 0;
            for(size_t localTransferIdx = 0; localTransferIdx < nTransfersPerElement; ++localTransferIdx) {
                auto &transfer = elementTransfers[localTransferIdx];
                size_t transferIdx = srcIdx*nTransfersPerElement + localTransferIdx; // global transfer idx
                size_t src = addressSrc + transfer.address;
                size_t transferSize = transfer.size;
                // transfer.firing - firing offset within element
                uint16 transferLastFiring = elementFirstFiring + transfer.firing;

                std::function<void()> onNewDataCallback;
                bool isLastTransfer = localTransferIdx == nTransfersPerElement-1;

                std::function<void()> callback;
                if(isLastTransfer) {
                    switch(strategy) {
                        case 0: callback = ON_NEW_DATA_CALLBACK(true, 0); break;
                        case 1: callback = ON_NEW_DATA_CALLBACK(true, 1); break;
                        case 2: callback = ON_NEW_DATA_CALLBACK(true, 2); break;
                        default: throw std::runtime_error("Unknown registrar strategy");
                    }
                }
                else {
                    switch(strategy) {
                        case 0: callback = ON_NEW_DATA_CALLBACK(false, 0); break;
                        case 1: callback = ON_NEW_DATA_CALLBACK(false, 1); break;
                        case 2: callback = ON_NEW_DATA_CALLBACK(false, 2); break;
                        default: throw std::runtime_error("Unknown registrar strategy");
                    }
                }
                ius4oem->ScheduleTransferRXBufferToHost(transferLastFiring, transferIdx, onNewDataCallback);

                transferFirstFiring = transferLastFiring + 1;
            }
            elementFirstFiring = elementLastFiring + 1;
        }
    }


private:
    Logger::Handle logger;
    Us4ROutputBuffer *dstBuffer{nullptr};
    const Us4OEMBuffer *srcBuffer{nullptr};
    Us4OEMImplBase *us4oem{nullptr};
    // All derived parameters
    IUs4OEM *ius4oem{nullptr};
    Ordinal us4oemOrdinal{0};
    std::vector<Transfer> elementTransfers;
    size_t srcNElements{0};
    size_t dstNElements{0};
    size_t nTransfersPerElement{0};
    // Number of transfer src points.
    size_t srcNTransfers{0};
    // Number of transfer dst points.
    size_t dstNTransfers{0};
    bool reprogramTransfers{false};
    int strategy{0};
    bool triggerOnRelease{false};
};


}

#endif //ARRUS_CORE_DEVICES_US4R_PROBEADAPTER_US4OEMDATATRANSFERREGISTRAR_H
