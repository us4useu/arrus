#ifndef ARRUS_ARRUS_CORE_DEVICES_US4R_US4RSUBSEQUENCE_H
#define ARRUS_ARRUS_CORE_DEVICES_US4R_US4RSUBSEQUENCE_H

#include <cstdint>
#include <optional>
#include <unordered_map>
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
#include <iostream>

namespace arrus::devices {

/**
 * Us4R TX/RX sub-sequence parameters.
 * Stores all the information required to apply sub-sequence on each us4OEM properly.
 */
class Us4RSubsequence {

public:
    Us4RSubsequence(std::vector<uint16_t> entries, uint32_t timeToNextTrigger,
                    const std::vector<Us4OEMBufferArrayDef> &arrays, FrameChannelMappingBuilder fcm)
        : entries(std::move(entries)), timeToNextTrigger(timeToNextTrigger), arrays(arrays),
          fcm(std::move(fcm)) {}

    /** The list of the physical firings (sequencer entries) to be executed, in the order of execution. */
    const std::vector<uint16_t> &getEntries() const { return entries; }
    /** The first physical firing of this sub-sequence (0 for an empty sub-sequence). */
    uint16_t getStart() const { return entries.empty() ? uint16_t(0) : entries.front(); }
    /** The last physical firing of this sub-sequence + 1 (0 for an empty sub-sequence).
     *  NOTE: the sub-sequence firings do not have to be consecutive, see getEntries. */
    uint16_t getEnd() const { return entries.empty() ? uint16_t(0) : ARRUS_SAFE_CAST(entries.back()+1, uint16_t); }
    uint32_t getTimeToNextTrigger() const { return timeToNextTrigger; }
    const std::vector<Us4OEMBufferArrayDef> &getArrayDefs() const { return arrays; }
    /** NOTE: this method builds a new FCM everytime is called */
    FrameChannelMappingImpl::Handle buildFCM() const { return fcm.build(); }
    bool empty() const { return entries.empty(); }

private:
    /** Physical firings (sequencer entries) to be executed, in the order of execution. */
    std::vector<uint16_t> entries;
    uint32_t timeToNextTrigger;
    /** Arrays for each OEM, TX/RX sequence id -> array. */
    std::vector<Us4OEMBufferArrayDef> arrays;
    /** FCM for the selected sub-sequence */
    FrameChannelMappingBuilder fcm;
};


class Us4RSubsequenceFactory {
public:
    /**
     * NOTE: regarding the mapping parameter start and end are assumed to be local per sequence,
     * i.e. logicalToPhysicalMapping.at(i).at(0) is counted from te beginning of the i-th sequence.
     *
     * @param mapping logical to physical TX/RX mapping
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
        this->logicalToPhysicalOpLocal = mapping;
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
                    ARRUS_SAFE_CAST(oemSequence.size(), uint16_t),
                    buffer.getArrayDef(seqId).getParts()
                );
            }
            this->opToNextFrame.push_back(seqMapping);
        }
    }

    /**
     * Creates the sub-sequence consisting of the [start, end) logical TX/RXs of the given sequence.
     */
    Us4RSubsequence get(SequenceId sequenceId, uint16_t start, uint16_t end, std::optional<float> sri) {
        ARRUS_REQUIRES_TRUE_IAE(start <= end, "Sub-sequence start should be not greater than the end");
        std::vector<uint16_t> ops;
        ops.reserve(static_cast<size_t>(end-start));
        for(uint16_t op = start; op < end; ++op) {
            ops.push_back(op);
        }
        return get(sequenceId, ops, sri);
    }

    /**
     * Creates the sub-sequence consisting of the given list of the logical TX/RXs of the given sequence.
     *
     * The TX/RXs do not have to be consecutive, however they must be provided in the increasing order
     * (the acquired data is stored in the output buffer in the TX/RX order).
     *
     * @param sequenceId the id of the TX/RX sequence to limit
     * @param ops the list of the logical TX/RXs (ordinal numbers) to run; empty means: turn off this sequence
     * @param sri the sequence repetition interval to apply
     */
    Us4RSubsequence get(SequenceId sequenceId, const std::vector<uint16_t> &ops, std::optional<float> sri) {
        if(ops.empty()) {
            // Return empty sub-sequence (i.e. the sequence should be turned off).
            std::vector<Us4OEMBufferArrayDef> arrays;
            for(const auto &buffer: oemBuffers) {
                const auto refArray = buffer.getArrayDef(sequenceId);
                arrays.push_back(Us4OEMBufferArrayDef{
                    refArray.getAddress(),
                    framework::NdArrayDef({0}, refArray.getDefinition().getDataType()),
                    {}
                });
            }
            return Us4RSubsequence{
                {}, 0, arrays, FrameChannelMappingBuilder{0, 0}
            };
        }
        validate(sequenceId, ops);
        // Determine the physical firings (sequencer entries) to be executed.
        // NOTE: a single logical TX/RX can be translated to more than one physical TX/RX (e.g. when the RX aperture
        // is larger than the number of the OEM RX channels).
        // entries: global (i.e. counted from the beginning of the sequencer table),
        // localEntries: local (i.e. counted from the beginning of the given TX/RX sequence).
        std::vector<uint16_t> entries, localEntries;
        for(const auto op: ops) {
            const auto [globalStart, globalEnd] = logicalToPhysicalOp.at(sequenceId).at(op);
            const auto [localStart, localEnd] = logicalToPhysicalOpLocal.at(sequenceId).at(op);
            for(uint16_t entry = globalStart; entry < globalEnd; ++entry) {
                entries.push_back(entry);
            }
            for(uint16_t entry = localStart; entry < localEnd; ++entry) {
                localEntries.push_back(entry);
            }
        }
        std::vector<Us4OEMBufferArrayDef> views;
        // Update us4OEM buffers.
        // We only limit the list of the parts and change the size and shape of the elements buffer (required
        // for creating new host buffer).
        // We do not recalculate firing numbers! This way transfer registrar will use the proper firing numbers.
        for (const auto &oemBuffer : oemBuffers) {
            views.push_back(getOEMBufferArrayDef(oemBuffer, sequenceId, localEntries));
        }
        // Update FCM.
        FrameChannelMappingBuilder outFCMBuilder = FrameChannelMappingBuilder::copy(*(fcm.at(sequenceId)));
        outFCMBuilder.select(ops);// keep the selected logical frames only
        // OEM nr -> number of frames
        std::vector<uint32> nFrames;
        for (size_t oem = 0; oem < oemBuffers.size(); ++oem) {
            // The frames not acquired by this sub-sequence are dropped, the remaining ones are renumbered
            // (e.g. the frames 3, 7 of the full sequence become the frames 0, 1 of the sub-sequence).
            auto frameNumbers = opToNextFrame.at(sequenceId).at(oem).getFrameNumbers(localEntries);
            nFrames.push_back(ARRUS_SAFE_CAST(frameNumbers.size(), uint32));
            if(!frameNumbers.empty()) {
                outFCMBuilder.remapPhysicalFrameNumbers((Ordinal)oem, frameNumbers);
            } // Otherwise there is no frame from the given OEM in FCM, so nothing to update.
        }
        // recalculate frame offsets
        outFCMBuilder.setNumberOfFrames(nFrames);
        outFCMBuilder.recalculateOffsets();
        return Us4RSubsequence{
            entries,
            getTimeToNextTrigger(sequenceId, localEntries, sri),
            views, outFCMBuilder
        };
    }

    /**
     * Re-creates OEM buffers based on the array definitions for each sequence.
     *
     * @param oemArrays TX/RX sequence -> OEM -> OEM Buffer array definition
     */
    std::vector<Us4OEMBuffer> recreateOEMBuffers(const std::vector<std::vector<Us4OEMBufferArrayDef>> &arrayDefs) {
        const auto nSequences = arrayDefs.size();
        const auto noems = oemBuffers.size();
        // OEM -> TX/RX sequence -> array definition (transposed arrayDefs)
        std::vector<std::vector<Us4OEMBufferArrayDef>> oemArrays(noems);

        for(size_t sequence = 0; sequence < nSequences; ++sequence) {
            for(size_t oem = 0; oem < noems; ++oem) {
                oemArrays.at(oem).push_back(arrayDefs.at(sequence).at(oem));
            }
        }
        std::vector<Us4OEMBuffer> result;
        for(size_t oem = 0; oem < oemBuffers.size(); ++oem) {
            const auto &buffer = oemBuffers.at(oem);
            // TX/RX sequence -> array def
            const auto &arrays = oemArrays.at(oem);
            std::vector<Us4OEMBufferElement> newElements;
            // Calculate the new element size.
            size_t newElementSize = std::accumulate(
                std::begin(arrays), std::end(arrays), size_t(0),
                [](const auto acc, const auto &array){
                    return acc + array.getSize();
                }
            );

            uint16 elementLastFiringView = 0;
            // Find the maximum number of the element firings.
            for(const auto &array: arrays) {
                for(const auto &part: array.getParts()) {
                    elementLastFiringView = std::max(elementLastFiringView, part.getEntryId());
                }
            }

            uint16 startFiring = 0;
            for(const auto &oldElement: buffer.getElements()) {
                newElements.emplace_back(
                    oldElement.getAddress(),
                    newElementSize,
                    oldElement.getGlobalFiring(),
                    ARRUS_SAFE_CAST(startFiring + elementLastFiringView, uint16) // sub-sequence last firing number
                );
                startFiring = oldElement.getGlobalFiring() + 1;
            }
            result.emplace_back(newElements, arrays);
        }
        return result;
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
                throw IllegalArgumentException("Accessing mapping outside the available range.");
            }
            return opToNextFrame.at(op);
        }

        /**
         * Returns the number of frames acquired by ops with numbers between [start, end) (right-side exclusive).
         */
        long getNumberOfFrames(uint16 start, uint16 end) {
            if(start == end) {
                return 0;
            }
            if(start > end || end > isRxOp.size()) {
                throw std::runtime_error("Accessing isRxOp outside the available range.");
            }
            long result = 0;
            for(uint16 i = start; i < end; ++i) {
                if(isRxOp.at(i)) {
                    ++result;
                }
            }
            return result;
        }

        /**
         * Returns the renumbering of the physical frames acquired by the given ops (firings):
         * the frame number in the full sequence -> the frame number in the sub-sequence.
         *
         * The ops that do not acquire any data are skipped; the remaining frames are numbered from 0,
         * in the order the ops are provided.
         */
        std::unordered_map<uint16, uint16> getFrameNumbers(const std::vector<uint16_t> &ops) {
            std::unordered_map<uint16, uint16> result;
            uint16 currentFrame = 0;
            for(const auto op: ops) {
                if(op >= isRxOp.size()) {
                    throw IllegalArgumentException("Accessing mapping outside the available range.");
                }
                if(isRxOp.at(op)) {
                    // NOTE: for the RX ops, opToNextFrame points to the frame acquired by this op.
                    result.emplace(opToNextFrame.at(op).value(), currentFrame++);
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
     * Converts input mapping (with the per-sequence local [start, end)) to the global TX/RX numbers.
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
                    return std::make_pair(ARRUS_SAFE_CAST(v.first + offset, uint16), ARRUS_SAFE_CAST(v.second + offset, uint16));
                }
            );
            result.push_back(newMap);
            if(!map.empty()) {
                // The physical end of the last TX/RX in the given sequence
                const auto &lastTxRx = (std::end(map)-1)->second;
                const auto sequenceSize = lastTxRx; // NOTE: end is the end of the range, exclusive
                offset += sequenceSize;
            }
        }
        return result;
    }

    void validate(SequenceId sequenceId, const std::vector<uint16_t> &ops) {
        if(sequenceId >= sequences.size()) {
            throw IllegalStateException(
                format("Sequence {} is outside of of the uploaded sequences (size: {})", sequenceId, sequences.size()));
        }
        const auto &seq = sequences.at(sequenceId);
        const auto currentSequenceSize = static_cast<uint16_t>(seq.getOps().size());
        for(size_t i = 0; i < ops.size(); ++i) {
            if(ops.at(i) >= currentSequenceSize) {
                throw IllegalArgumentException(
                    format("The TX/RX {} is outside of the scope of the sequence with id: {} [0, {})",
                           ops.at(i), sequenceId, currentSequenceSize));
            }
            if(i > 0 && ops.at(i) <= ops.at(i-1)) {
                throw IllegalArgumentException(
                    "The sub-sequence TX/RXs should be provided in the increasing order, without repetitions.");
            }
        }
    }

    /**
     * Returns the time to the next trigger (PRI) that should be set for the last firing of the sub-sequence,
     * so that the requested SRI is preserved.
     *
     * @param entries the physical firings (local to the given sequence) of the sub-sequence
     */
    unsigned int getTimeToNextTrigger(SequenceId sid, const std::vector<uint16_t> &entries,
                                      std::optional<float> sri) const {
        const auto &referenceOEMSequence = oemSequences.at(sid).at(0);
        auto lastOpPri = referenceOEMSequence.at(entries.back()).getPri();
        if(sri.has_value()) {
            // The total time of the sub-sequence (note: only the selected firings are executed).
            float totalPri = 0.0f;
            for(const auto entry: entries) {
                totalPri += referenceOEMSequence.at(entry).getPri();
            }
            if(totalPri >= sri.value()) {
                throw IllegalArgumentException(format("Sequence repetition interval {} cannot be set, "
                                                      "sequence total pri is equal {}", sri.value(), totalPri));
            }
            lastOpPri += sri.value() - totalPri;
        }
        return getPRIMicroseconds(lastOpPri);
    }

    /**
     * Returns the view of this buffer, limited to the given list of firings (parts) of the given array.
     *
     * @param entries the firings (local to the given sequence, i.e. the part numbers) to be kept
     */
    Us4OEMBufferArrayDef getOEMBufferArrayDef(const Us4OEMBuffer &buffer, ArrayId arrayId,
                                              const std::vector<uint16_t> &entries) const {
        const auto& arrayDef = buffer.getArrayDef(arrayId);
        if(entries.empty() || arrayDef.getSize() == 0) {
            // Empty the current (arrayDef) or the new (entries) array.
            return getEmptyArrayDef(arrayDef);
        }
        const auto& parts = arrayDef.getParts();
        Us4OEMBufferArrayParts newParts;
        newParts.reserve(entries.size());
        for(const auto entry: entries) {
            if(entry >= parts.size()) {
                throw IllegalArgumentException(
                    format("The index is outside of the scope of us4OEM Buffer view (index: {}, size: {})",
                           entry, parts.size()));
            }
            newParts.push_back(parts.at(entry));
        }
        // Calculate new shape of the array.
        auto oldShape = arrayDef.getDefinition().getShape();
        // Compute total number of samples acquired by this OEM
        unsigned newNSamples = std::accumulate(
            std::begin(newParts), std::end(newParts), 0,
            [](const auto &a, const auto &b){return a + b.getNSamples();});
        auto newShape = updateShape(oldShape, newNSamples);
        auto newDefinition = framework::NdArrayDef{newShape, arrayDef.getDefinition().getDataType()};
        // Calculate new address of the array.
        // The new address is the address of the first part of this sub-sequence.
        auto newAddress = std::begin(newParts)->getAddress();

        return Us4OEMBufferArrayDef {
            newAddress,
            newDefinition,
            newParts
        };
    }

    Us4OEMBufferArrayDef getEmptyArrayDef(const Us4OEMBufferArrayDef &refArrayDef) const {
        auto emptyArrayShape = refArrayDef.getDefinition().getShape();
        emptyArrayShape.getMutable(0) = 0;// The number of samples.
        return Us4OEMBufferArrayDef {refArrayDef.getAddress(),
            framework::NdArrayDef{emptyArrayShape, refArrayDef.getDefinition().getDataType()},
            {}
        };
    }

    static framework::NdArray::Shape updateShape(const framework::NdArray::Shape &currentShape, unsigned int totalNSamples) {
        if(totalNSamples == 0 || currentShape.empty()) { // Return empty array shape in case there are no samples acquired
            return {0,};
        }
        if(currentShape.size() != 2 && currentShape.size() != 3) {
            throw std::runtime_error("Illegal us4OEM output buffer element number of dimensions: " + std::to_string(currentShape.size()));
        }
        auto channelsAx = static_cast<uint32_t>(currentShape.size()-1);
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
    /** OEM buffers for the complete, input scheme, i.e. right after upload method was called. OEM ordinal -> buffer. */
    std::vector<Us4OEMBuffer> oemBuffers;
    /** Frame channel mappings for the complete, input scheme, i.e. right after upload method was called */
    std::vector<FrameChannelMappingImpl::Handle> fcm;
    /** sequence id -> op id -> GLOBAL firing start, end */
    std::vector<LogicalToPhysicalOp> logicalToPhysicalOp;
    /** sequence id -> op id -> Local (for the given sequence) firing start, stop */
    std::vector<LogicalToPhysicalOp> logicalToPhysicalOpLocal;
    /** sequence id -> OEM id -> op to next RF frame */
    std::vector<std::vector<OpToNextFrameMapping>> opToNextFrame;
};

}

#endif//ARRUS_ARRUS_CORE_DEVICES_US4R_US4RSUBSEQUENCE_H
