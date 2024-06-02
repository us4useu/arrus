#ifndef ARRUS_ARRUS_CORE_DEVICES_US4R_US4RSUBSEQUENCE_H
#define ARRUS_ARRUS_CORE_DEVICES_US4R_US4RSUBSEQUENCE_H

#include <cstdint>
#include <optional>

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
                    const std::vector<Us4OEMBuffer> &oemBuffers, const FrameChannelMappingBuilder &fcm)
        : start(start), end(anEnd), timeToNextTrigger(timeToNextTrigger), oemBuffers(oemBuffers),
          fcm(fcm) {}

    uint16_t getStart() const { return start; }
    uint16_t getEnd() const { return end; }
    uint32_t getTimeToNextTrigger() const { return timeToNextTrigger; }
    const std::vector<Us4OEMBuffer> &getOemBuffers() const { return oemBuffers; }
    /** NOTE: this method builds a new FCM everytime is called */
    FrameChannelMappingImpl::Handle buildFCM() const { return fcm.build(); }

private:
    /** Physical firing [start, end] NOTE: bothe inclusive */
    uint16_t start, end;
    uint32_t timeToNextTrigger;
    /** OEM buffers for the selected sub-sequence. */
    std::vector<Us4OEMBuffer> oemBuffers;
    /** FCM for the selected sub-sequence */
    FrameChannelMappingBuilder fcm;
};

class Us4RSubsequenceFactory {
public:

    Us4RSubsequenceFactory(
        const std::vector<LogicalToPhysicalOp> &mapping,
        const std::vector<::arrus::devices::us4r::TxRxParametersSequence> &masterSequences
    ) {
        this->logicalToPhysicalOp = createGlobalMapping(mapping);
        this->masterSequences = masterSequences;
    }


    Us4RSubsequence get(SequenceId sequenceId, uint16_t start, uint16_t end, const std::optional<float>& pri) {
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
            // TODO implement
            views.push_back(oemBuffer.getView(sequenceId, oemStart, oemEnd));
        }
        // Update FCM.
        FrameChannelMappingBuilder outFCMBuilder = FrameChannelMappingBuilder::copy(*(fcm.at(sequenceId)));
        outFCMBuilder.slice(start, end);// slice to logical frames to [start, end]
        // OEM nr -> number of frames
        std::vector<uint32> nFrames;
        for (size_t oem = 0; oem < fullSequenceOEMBuffers.size(); ++oem) {
            auto nextFrameNumber = physicalOpToNextFrame.at(oem).getNextFrame(oemStart);
            auto n = physicalOpToNextFrame.at(oem).getNumberOfFrames(oemStart, oemEnd);
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
            getTimeToNextTrigger(sequenceId, oemStart, oemEnd),
            views, outFCMBuilder
        };
    }

private:
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
            offset += map.size();
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

    unsigned int getTimeToNextTrigger(SequenceId sid, uint16 start, uint16 end) const {
        const auto &masterSeq = masterSequences.at(sid);
        auto sri = masterSeq.getSri();
        // NOTE: end is inclusive (and the below method expects [start, end) range.
        std::optional<float> lastPri = getSRIExtend(
            std::begin(masterSeq)+start,
            std::begin(masterSeq)+end+1,
            sri
        );
        if(lastPri.has_value()) {
            return getPRIMicroseconds(lastPri.value());
        }
        else {
            // Just use the PRI of the end TX/RX.
            return getPRIMicroseconds(masterSeq.at(end).getPri());
        }
    }

    /** TX/RX sequences from the  full scheme, i.e. right after upload method was called */
    std::vector<::arrus::ops::us4r::TxRxSequence> sequences;
    /** Master OEM sequence (with physical ops) */
    std::vector<::arrus::devices::us4r::TxRxParametersSequence> masterSequences;
    /** OEM buffers for the full scheme, i.e. right after upload method was called */
    std::vector<Us4OEMBuffer> oemBuffers;
    /** Frame channel mappings for the full scheme, i.e. right after upload method was called */
    std::vector<FrameChannelMappingImpl::Handle> fcm;
    ::arrus::ops::us4r::Scheme::WorkMode workMode;
    /** sequence, op id -> GLOBAL firing start, end */
    std::vector<LogicalToPhysicalOp> logicalToPhysicalOp;
};

}

#endif//ARRUS_ARRUS_CORE_DEVICES_US4R_US4RSUBSEQUENCE_H
