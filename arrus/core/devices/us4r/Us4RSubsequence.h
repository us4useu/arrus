#ifndef ARRUS_ARRUS_CORE_DEVICES_US4R_US4RSUBSEQUENCE_H
#define ARRUS_ARRUS_CORE_DEVICES_US4R_US4RSUBSEQUENCE_H

#include <cstdint>
#include <optional>
#include <utility>

#include "FrameChannelMappingImpl.h"
#include "arrus/common/format.h"
#include "arrus/core/api/common/exceptions.h"
#include "arrus/core/api/common/types.h"
#include "arrus/core/api/ops/us4r/TxRxSequence.h"
#include "arrus/core/devices/TxRxParameters.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMBuffer.h"
#include "arrus/core/devices/us4r/us4oem/utils.h"
#include "arrus/core/devices/us4r/types.h"

namespace arrus::devices {

/**
 * Us4R TX/RX sub-sequence parameters.
 * Stores all the information required to apply sub-sequence on each us4OEM properly.
 */
class Us4RSubsequence {

public:
    Us4RSubsequence(uint16_t start, uint16_t anEnd, uint32_t timeToNextTrigger,
                    const std::vector<Us4OEMBuffer> &oemBuffers, FrameChannelMappingBuilder fcm)
        : start(start), end(anEnd), timeToNextTrigger(timeToNextTrigger), oemBuffers(oemBuffers),
          fcm(std::move(fcm)) {}

    uint16_t getStart() const { return start; }
    uint16_t getEnd() const { return end; }
    uint32_t getTimeToNextTrigger() const { return timeToNextTrigger; }
    const std::vector<Us4OEMBuffer> &getOemBuffers() const { return oemBuffers; }
    /** NOTE: this method builds a new FCM everytime is called */
    FrameChannelMappingImpl::Handle buildFCM() const { return fcm.build(); }

private:
    /** Physical firing [start, end] NOTE: both inclusive */
    uint16_t start, end;
    uint32_t timeToNextTrigger;
    /** OEM buffers for the selected sub-sequence. */
    std::vector<Us4OEMBuffer> oemBuffers;
    /** FCM for the selected sub-sequence */
    FrameChannelMappingBuilder fcm;
};


class Us4RSubsequenceFactory {
public:
    /**
     * NOTE: regarding the mapping parameter start and end are assumed to be local per sequence,
     * i.e. logicalToPhysicalMapping.at(i).at(0) is counted from te beginning of the i-th sequence.
     *
     * @param mapping
     * @param oemSequences actual TX/RX sequences on each OEM; sequence id -> OEM -> op. NOTE: these are sequences after
     *   splitting RXs into subapertures (i.e. max 32 active elements)!
     * @param oemBuffers OEM buffers; OEM -> buffer
     * @param fcms frame channel mappings; sequence id -> FCM
     */
    Us4RSubsequenceFactory(
        const std::vector<::arrus::ops::us4r::TxRxSequence> &sequences,
        const std::vector<LogicalToPhysicalOp> &mapping,
        const std::vector<std::vector<::arrus::devices::us4r::TxRxParametersSequence>> &oemSequences,
        const std::vector<Us4OEMBuffer> &oemBuffers,
        const std::vector<FrameChannelMappingImpl::Handle> &fcms
    ) {
        this->sequences = sequences;
        this->logicalToPhysicalOp = createGlobalMapping(mapping);
        this->oemSequences = oemSequences;
        this->oemBuffers = oemBuffers;
        for(const auto &m: fcms) {
            this->fcm.emplace_back(
                FrameChannelMappingBuilder::copy(*m).build()
            );
        }
        // Create OpToNextFrameMappings
        for(size_t seqId = 0; seqId < this->oemSequences.size(); ++seqId) {
            const auto &oemSeq = this->oemSequences.at(seqId);
            std::vector<OpToNextFrameMapping> seqMapping;
            ARRUS_REQUIRES_EQUAL_IAE(this->oemBuffers.size(), oemSeq.size());
            for(size_t i = 0; i < oemSeq.size(); ++i) {
                const auto &oemSequence = oemSeq.at(i);
                const auto &buffer = this->oemBuffers.at(i);
                seqMapping.emplace_back(
                    oemSequence.size(),
                    buffer.getArrayDef(seqId).getParts()
                );
            }
            this->opToNextFrame.push_back(seqMapping);
        }
    }

    Us4RSubsequence get(SequenceId sequenceId, uint16_t start, uint16_t end, std::optional<float> sri) {
        validate(sequenceId, start, end);
        // physical [start, end]
        uint16_t oemStart = logicalToPhysicalOp.at(sequenceId).at(start).first;
        uint16_t oemEnd = logicalToPhysicalOp.at(sequenceId).at(end).second;

        std::vector<Us4OEMBuffer> views;
        // Update us4OEM buffers.
        // We only limit the range of the parts list and change the size and shape of the elements buffer (required
        // for creating new host buffer).
        // We do not recalculate firing numbers! This way transfer registrar will use the proper firing numbers.
        for (const auto &oemBuffer : oemBuffers) {
            views.push_back(getOEMBufferView(oemBuffer, sequenceId, oemStart, oemEnd));
        }
        // Update FCM.
        FrameChannelMappingBuilder outFCMBuilder = FrameChannelMappingBuilder::copy(*(fcm.at(sequenceId)));
        outFCMBuilder.slice(start, end);// slice to logical frames to [start, end]
        // OEM nr -> number of frames
        std::vector<uint32> nFrames;
        for (size_t oem = 0; oem < oemBuffers.size(); ++oem) {
            auto nextFrameNumber = opToNextFrame.at(sequenceId).at(oem).getNextFrame(oemStart);
            auto n = opToNextFrame.at(sequenceId).at(oem).getNumberOfFrames(oemStart, oemEnd);
            nFrames.push_back(n);
            if (nextFrameNumber.has_value()) {
                // Subtract from the physical frame numbers, the number of preceeding frames (e.g. move frame 3 to 0).
                outFCMBuilder.subtractPhysicalFrameNumber((Ordinal)oem, nextFrameNumber.value());
            } // Otherwise there is no frame from the given OEM in FCM, so nothing to update.
        }
        // recalculate frame offsets
        outFCMBuilder.setNumberOfFrames(nFrames);
        outFCMBuilder.recalculateOffsets();
        return Us4RSubsequence{
            oemStart, oemEnd,
            getTimeToNextTrigger(sequenceId, oemStart, oemEnd, sri),
            views, outFCMBuilder
        };
    }

private:
    struct OpToNextFrameMapping {

        OpToNextFrameMapping(uint16_t nFirings, const std::vector<Us4OEMBufferArrayPart> &frames) {
            std::optional<uint16_t> currentFrameNr = std::nullopt;
            opToNextFrame = std::vector<std::optional<uint16_t>>(nFirings, std::nullopt);
            isRxOp = std::vector<bool>(nFirings, false);
            for (int firing = nFirings - 1; firing >= 0; --firing) {
                const auto &frame = frames.at(firing);
                if (frame.getSize() > 0) {
                    if (!currentFrameNr.has_value()) {
                        currentFrameNr = (uint16_t)0;
                    } else {
                        currentFrameNr = static_cast<uint16_t>(currentFrameNr.value() + 1);
                    }
                    isRxOp.at(firing) = true;
                }
                opToNextFrame.at(firing) = currentFrameNr;
            }
            // Reverse the numbering.
            // e.g.
            // 0 -> 1, 1 -> 1, 2 -> 0, 3 -> 0
            // =>
            // 0 -> 0, 1 -> 0, 2 -> 1, 3 -> 1
            if (currentFrameNr.has_value()) {
                auto maxFrameNr = currentFrameNr.value();
                for (auto &nextFrame : opToNextFrame) {
                    if (nextFrame.has_value()) {
                        nextFrame.value() = maxFrameNr - nextFrame.value();
                    }
                }
            } // otherwise opToNextFrame is all of nullopts, nothing to update
        }

        std::optional<uint16> getNextFrame(uint16 op) {
            if(op >= opToNextFrame.size()) {
                throw IllegalArgumentException("Accessing mapping outside the avialable range.");
            }
            return opToNextFrame.at(op);
        }

        /**
         * Returns the number of frames acquired by ops with numbers between [start, end] (both inclusive).
         */
        long getNumberOfFrames(uint16 start, uint16 end) {
            if(start > end || end >= isRxOp.size()) {
                throw std::runtime_error("Accessing isRxOp outside the available range.");
            }
            long result = 0;
            for(uint16 i = start; i <= end; ++i) {
                if(isRxOp.at(i)) {
                    ++result;
                }
            }
            return result;
        }
        // op (firing) number -> next frame number, relative to the full sequence.
        std::vector<std::optional<uint16_t>> opToNextFrame;
        // op (firing) number -> whether there is some data acquisition done by this op
        std::vector<bool> isRxOp;
    };


    /**
     * Converts input mapping (with the per-sequence local [start, end] to the global TX/RX numbers.
     */
    std::vector<LogicalToPhysicalOp> createGlobalMapping(const std::vector<LogicalToPhysicalOp> &localMap) {
        // NOTE: ASSUMING that subsequence OEMs are programmed sequentially, and there are no gaps, etc.
        // between subsequent ops.
        std::vector<LogicalToPhysicalOp> result;
        size_t offset = 0;
        for(const auto &map: localMap) {
            LogicalToPhysicalOp newMap(map.size());
            std::transform(
                std::begin(map), std::end(map), std::begin(newMap),
                [offset](const std::pair<OpId, OpId> &v) {
                    return std::make_pair(v.first + offset, v.second + offset);
                }
            );
            result.push_back(newMap);
            if(!map.empty()) {
                // The physical end of the last TX/RX
                const auto &lastTxRx = (std::end(map)-1)->second;
                const auto sequenceSize = lastTxRx+1; // NOTE: end is the end of the range, inclusive
                offset += sequenceSize;
            }
        }
        return result;
    }

    void validate(SequenceId sequenceId, uint16 start, uint16 end) {
        if(sequenceId >= sequences.size()) {
            throw IllegalStateException(
                format("Sequence {} is outside of of the uploaded sequences (size: {})", sequenceId, sequences.size()));
        }
        const auto &seq = sequences.at(sequenceId);
        const auto currentSequenceSize = static_cast<uint16_t>(seq.getOps().size());
        if(end >= currentSequenceSize) {
            throw IllegalArgumentException(
                format("The new sub-sequence [{}, {}] is outside of the scope of the sequence with id: {} "
                             " [0, {})", start, end, sequenceId, currentSequenceSize));
        }
    }

    unsigned int getTimeToNextTrigger(SequenceId sid, uint16 start, uint16 end, std::optional<float> sri) const {
        const auto &referenceOEMSequence = oemSequences.at(sid).at(0);
        // NOTE: end is inclusive (and the below method expects [start, end) range.
        std::optional<float> lastPri = getSRIExtend(
            std::begin(referenceOEMSequence)+start,
            std::begin(referenceOEMSequence)+end+1,
            sri
        );
        if(lastPri.has_value()) {
            return getPRIMicroseconds(lastPri.value());
        }
        else {
            // Just use the PRI of the end TX/RX.
            return getPRIMicroseconds(referenceOEMSequence.at(end).getPri());
        }
    }

    /**
     * Returns the view of this buffer for slice [start, end] (note: end is inclusive) of the given array.
     */
    Us4OEMBuffer getOEMBufferView(const Us4OEMBuffer &buffer, ArrayId arrayId, uint16 start, uint16 end) const {
        if (start > end) {
            throw IllegalArgumentException("Us4OEMBufferView: start cannot exceed end");
        }
        const auto& arrayDef = buffer.getArrayDef(arrayId);
        const auto& parts = arrayDef.getParts();

        if (end >= parts.size()) {
            throw IllegalArgumentException(
                format("The index is outside of the scope of us4OEM Buffer view (index: {}, size: {})",
                       end, parts.size()));
        }
        auto b = std::begin(parts);
        // NEW ARRAY DEF (A SINGLE ARRAY SHOULD BE DEFINED)
        Us4OEMBufferArrayParts newParts(b+start, b+end+1); // NOTE: +1 because end is inclusive
        // Calculate new shape of the array.
        auto oldShape = arrayDef.getDefinition().getShape();
        // Compute total number of samples acquired by this OEM
        unsigned newNSamples = std::accumulate(
            std::begin(newParts), std::end(newParts), 0,
            [](const auto &a, const auto &b){return a + b.getNSamples();});
        auto newShape = updateShape(oldShape, newNSamples);
        auto newDefinition = framework::NdArrayDef{newShape, arrayDef.getDefinition().getDataType()};
        // Calculate new address of the array.
        // The new address is the current address + offset caused by the start part.
        auto newAddress = arrayDef.getAddress() + std::begin(newParts)->getAddress();
        Us4OEMBufferArrayDef newArrayDef{
            newAddress,
            newDefinition,
            newParts
        };
        // NEW ELEMENTS -- RECALCULATE ELEMENT SIZE.
        std::vector<Us4OEMBufferElement> newElements;
        for(const auto &oldElement: buffer.getElements()) {
            newElements.emplace_back(
                oldElement.getAddress() + newArrayDef.getAddress(),
                newArrayDef.getSize(),
                oldElement.getGlobalFiring()
            );
        }
        return Us4OEMBuffer(newElements, {newArrayDef});
    }

    static framework::NdArray::Shape updateShape(const framework::NdArray::Shape &currentShape, unsigned int totalNSamples) {
        if(currentShape.size() != 2 && currentShape.size() != 3) {
            throw std::runtime_error("Illegal us4OEM output buffer element shape order: " + std::to_string(currentShape.size()));
        }
        auto channelsAx = currentShape.size() == 0 ? uint32_t(0) : static_cast<uint32_t>(currentShape.size()-1);
        bool isDDCOn = currentShape.size() == 3;
        auto nChannels = static_cast<uint32_t>(currentShape.get(channelsAx));
        if(isDDCOn) {
            return {totalNSamples, 2, nChannels};
        } else {
            return {totalNSamples, nChannels};
        }
    }

    /** TX/RX sequences from the complete, input scheme, i.e. right after upload method was called */
    std::vector<::arrus::ops::us4r::TxRxSequence> sequences;
    /** OEM sequences (with physical ops) */
    std::vector<std::vector<::arrus::devices::us4r::TxRxParametersSequence>> oemSequences;
    /** OEM buffers for the complete, input scheme, i.e. right after upload method was called */
    std::vector<Us4OEMBuffer> oemBuffers;
    /** Frame channel mappings for the complete, input scheme, i.e. right after upload method was called */
    std::vector<FrameChannelMappingImpl::Handle> fcm;
    /** sequence id -> op id -> GLOBAL firing start, end */
    std::vector<LogicalToPhysicalOp> logicalToPhysicalOp;
    /** sequence id -> OEM id -> op to next RF frame */
    std::vector<std::vector<OpToNextFrameMapping>> opToNextFrame;
};

}

#endif//ARRUS_ARRUS_CORE_DEVICES_US4R_US4RSUBSEQUENCE_H
