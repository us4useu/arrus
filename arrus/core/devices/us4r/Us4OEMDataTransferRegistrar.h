#ifndef ARRUS_CORE_DEVICES_US4R_US4OEMDATATRANSFERREGISTRAR_H
#define ARRUS_CORE_DEVICES_US4R_US4OEMDATATRANSFERREGISTRAR_H

#include "arrus/common/compiler.h"
#include "arrus/core/api/common/types.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/devices/us4r/Us4ROutputBuffer.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMImpl.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMImplBase.h"

namespace arrus::devices {

class Transfer {
public:
    Transfer(size_t destination, size_t source, size_t size, uint16 firing)
        : destination(destination), source(source), size(size), firing(firing) {}

    bool operator==(const Transfer &rhs) const {
        return source == rhs.source && destination == rhs.destination && size == rhs.size && firing == rhs.firing;
    }
    bool operator!=(const Transfer &rhs) const { return !(rhs == *this); }

    /** Destination address, relative to the beginning of the element. */
    size_t destination{0};
    /** Source address, relative to the beginning of the element. */
    size_t source{0};
    size_t size{0};
    uint16 firing{0};
};

/**
 * Registers transfers from us4R internal DDR memory to the destination (host) memory.
 */
class Us4OEMDataTransferRegistrar {
public:
    using ArrayTransfers = std::vector<Transfer>;
    static constexpr size_t MAX_N_TRANSFERS = 256;
    static constexpr size_t MAX_TRANSFER_SIZE = Us4OEMImpl::MAX_TRANSFER_SIZE;

    Us4OEMDataTransferRegistrar(Us4ROutputBuffer *dst, const Us4OEMBuffer &src, Us4OEMImplBase *us4oem)
        : logger(loggerFactory->getLogger()), dstBuffer(dst), srcBuffer(src) {
        ARRUS_INIT_COMPONENT_LOGGER(logger, "Us4OEMDataTransferRegistrar");
        if (dst->getNumberOfElements() % src.getNumberOfElements() != 0) {
            throw IllegalArgumentException("Host buffer should have multiple of rx buffer elements.");
        }
        ius4oem = us4oem->getIUs4OEM();
        us4oemOrdinal = us4oem->getDeviceId().getOrdinal();
        elementTransfers = createTransfers(dst, src, us4oem->getDeviceId().getOrdinal());

        srcNElements = src.getNumberOfElements();
        dstNElements = dst->getNumberOfElements();
        nTransfersPerElement = getNumberOfTransfers();
        // Number of transfer src points.
        srcNTransfers = nTransfersPerElement * srcNElements;// Should be <= 256
        // Number of transfer dst points.
        dstNTransfers = nTransfersPerElement * dstNElements;// Can be > 256

        ARRUS_REQUIRES_AT_MOST(srcNTransfers, MAX_N_TRANSFERS, "Exceeded maximum number of transfers.");

        // If true: create only nSrc transfers, the callback function will reprogram the appropriate number transfers.
        if (dstNTransfers > MAX_N_TRANSFERS) {
            strategy = 2;
        } else if (dstNTransfers > srcNTransfers) {
            // reschedule needed
            strategy = 1;
        } else {
            // nTransferDst == nTransferSrc
            strategy = 0;
        }
        this->logger->log(LogSeverity::DEBUG,
                          format("Us4OEM:{}, transfer strategy: {}", us4oem->getDeviceId().getOrdinal(), strategy));
    }

    [[nodiscard]] size_t getNumberOfTransfers() const {
        return std::accumulate(std::begin(elementTransfers), std::end(elementTransfers), 0,
                               [](const auto &a, const auto &b) { return a + b.size(); });
    }

    void registerTransfers() {
        // Page-lock all host dst points.
        pageLockDstMemory();

        // Send page descriptors to us4OEM DMA.
        size_t nSrcPoints = srcNElements;
        size_t nDstPoints = strategy == 2 ? srcNElements : dstNElements;

        programTransfers(nSrcPoints, nDstPoints);
        scheduleTransfers();
    }

    void unregisterTransfers(bool cleanupSequencer = false) {
        pageUnlockDstMemory();
        if(cleanupSequencer) {
            cleanupSequencerTransfers();
        }
    }

    void cleanupSequencerTransfers() {
        uint16 elementFirstFiring = 0;
        for(uint16 srcIdx = 0; srcIdx < srcNElements; ++srcIdx) {
            for(auto &arrayTransfers: elementTransfers) {
                for(const auto &transfer: arrayTransfers) {
                    auto firing = elementFirstFiring + transfer.firing;
                    ius4oem->ClearTransferRXBufferToHost(firing);
                }
            }
            // element.getFiring() -- the last firing of the given element
            elementFirstFiring = srcBuffer.getElement(srcIdx).getFiring() + 1;
        }
    }

    /**
     * Creates for each array to be produced by the given OEM.
     * NOTE: this method returns source/destination addresses relative to the beginning of each element
     * (we assume that each element has exactly the same transfer layout).
     *
     * The ArrayTransfers will be empty if a given OEM does not produce data for the selected array.
     *
     * This method required that each array part has size <= MAX_TRANSFER_SIZE.
     */
    static std::vector<ArrayTransfers> createTransfers(
        const Us4ROutputBuffer *dst, const Us4OEMBuffer &src, Ordinal oem) {
        std::vector<ArrayTransfers> result;

        for (ArrayId arrayId = 0; arrayId < src.getNumberOfArrays(); ++arrayId) {
            // Determines transfers for the part of this array, produced by the given OEM.
            ArrayTransfers transfers;
            auto &parts = src.getParts(arrayId);
            if (parts.empty()) {
                // this OEM does not produce data for this array
                result.push_back(transfers);
            } else {
                // This OEM produces some data for this array.
                size_t source = src.getArrayAddressRelative(arrayId);
                size_t destination = dst->getArrayAddressRelative(arrayId, oem);
                size_t size = 0;
                uint16 firing = parts[0].getEntryId();// the firing that finishes given transfer
                for (auto &part : parts) {
                    ARRUS_REQUIRES_TRUE_E(part.getSize() <= MAX_TRANSFER_SIZE,
                                          ArrusException(format("A single frame cannot exceed {} bytes, got: {}",
                                                                part.getSize(), MAX_TRANSFER_SIZE)));

                    if (size + part.getSize() > MAX_TRANSFER_SIZE) {
                        transfers.emplace_back(destination, source, size, firing);
                        source = part.getAddress();
                        destination += size;
                        size = 0;
                    }
                    size += part.getSize();
                    firing = part.getEntryId();
                }
                if (size > 0) {
                    transfers.emplace_back(destination, source, size, firing);
                }
                result.emplace_back(transfers);
            }
        }
        return result;
    }

    void pageLockDstMemory() {
        for (uint16 dstIdx = 0, srcIdx = 0; dstIdx < dstNElements; ++dstIdx, srcIdx = (srcIdx + 1) % srcNElements) {
            uint8 *addressDst = dstBuffer->getAddress(dstIdx);
            // NOTE: addressSrc should be the address of the complete buffer element here -- even if
            // we are processing some sub-sequence buffer here (i.e. setSubsequence was used).
            // The reason for that is that the transfer src address is relative to the begining of the FULL buffer
            // element (because element parts are relative to the FULL element).
            size_t addressSrc = srcBuffer.getElement(srcIdx).getAddress();// byte-addressed
            for (const auto &arrayTransfers : elementTransfers) {
                for (auto &transfer : arrayTransfers) {
                    uint8 *dst = addressDst + transfer.destination;
                    size_t src = addressSrc + transfer.source;
                    size_t size = transfer.size;
                    ius4oem->PrepareHostBuffer(dst, size, src, false);
                }
            }
        }
    }

    void pageUnlockDstMemory() {
        for (uint16 dstIdx = 0, srcIdx = 0; dstIdx < dstNElements; ++dstIdx, srcIdx = (srcIdx + 1) % srcNElements) {
            uint8 *addressDst = dstBuffer->getAddressUnsafe(dstIdx);
            size_t addressSrc = srcBuffer.getElement(srcIdx).getAddress();// byte-addressed
            for (const auto &arrayTransfers : elementTransfers) {
                for (auto &transfer : arrayTransfers) {
                    uint8 *dst = addressDst + transfer.destination;
                    size_t src = addressSrc + transfer.source;
                    size_t size = transfer.size;
                    ius4oem->ReleaseTransferRxBufferToHost(dst, size, src);
                }
            }
        }
    }

    void programTransfers(size_t nSrcPoints, size_t nDstPoints) {
        size_t transferIdx = 0;// global transfer idx
        for (uint16 dstIdx = 0, srcIdx = 0; dstIdx < nDstPoints; ++dstIdx, srcIdx = (srcIdx + 1) % nSrcPoints) {
            uint8 *addressDst = dstBuffer->getAddress(dstIdx);
            size_t addressSrc = srcBuffer.getElement(srcIdx).getAddress();// byte-addressed
            for (const auto &arrayTransfers : elementTransfers) {
                for (size_t localIdx = 0; localIdx < arrayTransfers.size(); ++localIdx, ++transferIdx) {
                    auto &transfer = arrayTransfers[localIdx];
                    uint8 *dst = addressDst + transfer.destination;
                    size_t src = addressSrc + transfer.source;
                    size_t size = transfer.size;
                    ius4oem->PrepareTransferRXBufferToHost(transferIdx, dst, size, src, false);
                }
            }
        }
    }

// ON NEW DATA CALLBACK POLICIES
// TODO replace macros with templates after refactoring us4r-api
#define ARRUS_ON_NEW_DATA_CALLBACK_signal_true                                                                         \
    dstBuffer->signal(us4oemOrdinal, currentDstIdx);                                                                   \
    currentDstIdx = (int16) ((currentDstIdx + srcNElements) % dstNElements);

#define ARRUS_ON_NEW_DATA_CALLBACK_signal_false

// Strategy 0: keep transfers as they are (nSrc == nDst)
#define ARRUS_ON_NEW_DATA_CALLBACK_strategy_0

// Strategy 1: change sequencer firings definition, so the next firing will trigger the next portion of transfers
// (nSrc < nDst && nDst <= 256)
#define ARRUS_ON_NEW_DATA_CALLBACK_strategy_1                                                                          \
    currentTransferIdx = (int16) ((currentTransferIdx + srcNTransfers) % dstNTransfers);                               \
    ius4oem->ScheduleTransferRXBufferToHost(transferLastFiring, currentTransferIdx, nullptr);

// Strategy 2: re-program transfer, so in the next call this transfer will write to subsequent dst element
// (nDst > 256)
#define ARRUS_ON_NEW_DATA_CALLBACK_strategy_2                                                                          \
    uint16 nextElementIdx = (int16) ((currentDstIdx + srcNElements) % dstNElements);                                   \
    auto nextDstAddress = dstBuffer->getAddress(nextElementIdx);                                                       \
    nextDstAddress += transfer.destination;                                                                            \
    ius4oem->PrepareTransferRXBufferToHost(currentTransferIdx, nextDstAddress, transferSize, src, false);

#define ARRUS_ON_NEW_DATA_CALLBACK(signal, strategy)                                                                   \
    [=, currentDstIdx = srcIdx, currentTransferIdx = transferIdx]() mutable {                                          \
        IGNORE_UNUSED(currentTransferIdx);                                                                             \
        IGNORE_UNUSED(currentDstIdx);                                                                                  \
        try {                                                                                                          \
            ARRUS_ON_NEW_DATA_CALLBACK_strategy_##strategy ARRUS_ON_NEW_DATA_CALLBACK_signal_##signal                  \
        } catch (const std::exception &e) {                                                                            \
            logger->log(LogSeverity::ERROR, format("Us4OEM {}: callback exception: {}", us4oemOrdinal, e.what()));     \
        } catch (...) {                                                                                                \
            logger->log(LogSeverity::ERROR, format("Us4OEM {}: callback unknown exception.", us4oemOrdinal));          \
        }                                                                                                              \
    }

    void scheduleTransfers() {
        // Schedule transfers only from the start points (nSrc calls), dst pointers will be incremented
        // appropriately (if necessary).
        size_t transferIdx = 0;       // global transfer idx
        uint16 elementFirstFiring = 0;// NOTE: global, counted from 0
        for (int16 srcIdx = 0; srcIdx < int16(srcNElements); ++srcIdx) {
            const auto &element = srcBuffer.getElement(srcIdx);
            size_t addressSrc = element.getAddress();// bytes addressed
            uint16 elementLastFiring = element.getFiring();
            // for each element's part transfer:
            size_t localIdx = 0;
            for (const auto &arrayTransfers : elementTransfers) {
                for(const auto &transfer: arrayTransfers) {
                    size_t src = addressSrc + transfer.source;// used by callback strategy 2
                    size_t transferSize = transfer.size;
                    // transfer.firing - firing offset within element
                    uint16 transferLastFiring = elementFirstFiring + transfer.firing;

                    bool isLastTransfer = localIdx == nTransfersPerElement - 1;
                    std::function<void()> callback;
                    if (isLastTransfer) {
                        switch (strategy) {
                        case 0: callback = ARRUS_ON_NEW_DATA_CALLBACK(true, 0); break;
                        case 1: callback = ARRUS_ON_NEW_DATA_CALLBACK(true, 1); break;
                        case 2: callback = ARRUS_ON_NEW_DATA_CALLBACK(true, 2); break;
                        default: throw std::runtime_error("Unknown us4R buffer registrar strategy");
                        }
                    } else {
                        switch (strategy) {
                        case 0: callback = ARRUS_ON_NEW_DATA_CALLBACK(false, 0); break;
                        case 1: callback = ARRUS_ON_NEW_DATA_CALLBACK(false, 1); break;
                        case 2: callback = ARRUS_ON_NEW_DATA_CALLBACK(false, 2); break;
                        default: throw std::runtime_error("Unknown us4R buffer registrar strategy");
                        }
                    }
                    ius4oem->ScheduleTransferRXBufferToHost(transferLastFiring, transferIdx, callback);
                    ++localIdx; ++transferIdx;
                }
            }
            elementFirstFiring = elementLastFiring + 1;
        }
    }

private:
    Logger::Handle logger;
    Us4ROutputBuffer *dstBuffer{nullptr};
    const Us4OEMBuffer srcBuffer;
    // All derived parameters
    IUs4OEM *ius4oem{nullptr};
    Ordinal us4oemOrdinal{0};
    /** The aray of each transfer. NOTE: all addresses are relative to the beginning of the buffer element! **/
    std::vector<ArrayTransfers> elementTransfers;
    size_t srcNElements{0};
    size_t dstNElements{0};
    size_t nTransfersPerElement{0};
    // Number of transfer src points.
    size_t srcNTransfers{0};
    // Number of transfer dst points.
    size_t dstNTransfers{0};
    int strategy{0};
};

}// namespace arrus::devices

#endif//ARRUS_CORE_DEVICES_US4R_US4OEMDATATRANSFERREGISTRAR_H
