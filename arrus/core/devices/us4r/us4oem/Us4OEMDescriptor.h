#ifndef ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMLIMITS_H
#define ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMLIMITS_H

#include "arrus/core/api/common/Interval.h"
#include "arrus/core/api/common/types.h"
#include "arrus/core/api/devices/us4r/RxSettings.h"
#include "arrus/core/api/ops/us4r/constraints/TxRxSequenceLimits.h"
#include <ius4oem.h>

#include <utility>
#include <vector>

namespace arrus::devices {

class Us4OEMDescriptorBuilder;

/**
 * Location of an addressable (physical) us4OEM channel in the acquired RX data.
 * slot: RX slot (FPGA RX channel mapping value), row: raw row in each group of rxInterleave raw rows (ADC clocks),
 * group: the RX mux setting the channel requires (channels of different groups cannot be acquired together
 * when they share a slot).
 */
struct RxInputLocation {
    ChannelIdx slot;
    ChannelIdx row;
    ChannelIdx group;
};

/** Physical channel -> (RX slot, raw row). Empty: the default slot = channel % nRxChannels, row = 0. */
using RxInputTable = std::vector<std::pair<uint8_t, uint8_t>>;

inline RxInputLocation getRxInputLocation(ChannelIdx physicalChannel, ChannelIdx nRxChannels, uint32_t rxInterleave,
                                          const RxInputTable &table) {
    const auto group = static_cast<ChannelIdx>(physicalChannel / (nRxChannels * rxInterleave));
    if (table.empty()) {
        if (rxInterleave > 1) {
            throw IllegalArgumentException("RX input table is required for interleaved RX.");
        }
        return RxInputLocation{static_cast<ChannelIdx>(physicalChannel % nRxChannels), 0, group};
    }
    const auto &[slot, row] = table.at(physicalChannel);
    return RxInputLocation{slot, row, group};
}

/**
 * Us4OEM parameters and constraints.
 *
 * This data object is intended to store all the necessary information about
 * the OEM (the number of channels, etc).
 */
class Us4OEMDescriptor {
public:
    static constexpr ChannelIdx N_TX_CHANNELS = IUs4OEM::NCH;
    static constexpr ChannelIdx N_ADDR_CHANNELS = N_TX_CHANNELS;
    static constexpr ChannelIdx ACTIVE_CHANNEL_GROUP_SIZE = 8;
    static constexpr ChannelIdx N_ACTIVE_CHANNEL_GROUPS = N_TX_CHANNELS / ACTIVE_CHANNEL_GROUP_SIZE;
    // TODO deprecated! please use the getNRxChannels method
    static constexpr ChannelIdx N_RX_CHANNELS = 32;
    // TODO deprecated!
    static constexpr size_t TGC_N_SAMPLES = 1022;
    static constexpr unsigned MAX_IRQ_NR = IUs4OEM::MAX_IRQ_NR;
    uint32_t US4OEM_LEGACY_REVISION = 1;
    uint32_t US4OEM_PLUS_REVISION = 2;


    Us4OEMDescriptor(
        uint32_t version, ChannelIdx nRxChannels, float minRxTime, float rxTimeEpsilon, float sequenceReprogrammingTime,
        float samplingFrequency, size_t ddrSize, size_t maxTransferSize, float nPeriodsResolution,
        bool master, ops::us4r::TxRxSequenceLimits txRxSequenceLimits, uint8_t nTimeouts, uint32_t sampleTxStart,
        uint32_t rxInterleave = 1, RxInputTable rxInputTable = {})
        : version(version), nRxChannels(nRxChannels), minRxTime(minRxTime), rxTimeEpsilon(rxTimeEpsilon),
          sequenceReprogrammingTime(sequenceReprogrammingTime), samplingFrequency(samplingFrequency), ddrSize(ddrSize),
          maxTransferSize(maxTransferSize), nPeriodsResolution(nPeriodsResolution),
          master(master), txRxSequenceLimits(std::move(txRxSequenceLimits)), nTimeouts(nTimeouts),
          sampleTxStart(sampleTxStart), rxInterleave(rxInterleave), rxInputTable(std::move(rxInputTable)) {}

    bool isUs4OEMLegacy() {return version == US4OEM_LEGACY_REVISION; }
    bool isUs4OEMPlus() {return version >= US4OEM_PLUS_REVISION; }
    ChannelIdx getNTxChannels() const { return N_TX_CHANNELS; }
    ChannelIdx getNRxChannels() const { return nRxChannels; }
    ChannelIdx getNAddressableRxChannels() const { return N_ADDR_CHANNELS; }
    ChannelIdx getNActiveChannelGroups() const {return N_ACTIVE_CHANNEL_GROUPS; }
    ChannelIdx getActiveChannelGroupSize() const { return ACTIVE_CHANNEL_GROUP_SIZE; }
    float getMinRxTime() const { return minRxTime; }
    float getRxTimeEpsilon() const { return rxTimeEpsilon; }
    float getSequenceReprogrammingTime() const { return sequenceReprogrammingTime; }
    float getSamplingFrequency() const { return samplingFrequency; }
    size_t getDdrSize() const { return ddrSize; }
    size_t getMaxTransferSize() const { return maxTransferSize; }
    const ops::us4r::TxRxSequenceLimits &getTxRxSequenceLimits() const { return txRxSequenceLimits; }
    float getNPeriodsResolution() const { return nPeriodsResolution; }
    bool isMaster() const { return master; }
    unsigned getMaxIRQNumber() {return MAX_IRQ_NR; }
    uint8_t getNTimeouts() const { return nTimeouts; }
    /** Returns the RX sample number (relative to the trigger), when the TX delay = 0 time occurs. */
    uint32_t getSampleTxStart() const { return sampleTxStart; }
    /**
     * Number of AFE inputs time-multiplexed into a single RX slot (AFE58JD32: 2, odd/even inputs on alternate
     * ADC clocks; otherwise 1). getNRxChannels() is the number of RX slots (FPGA RX channel mapping size),
     * getSamplingFrequency() is the per-input sampling frequency, i.e. ADC clock / rxInterleave.
     * The RX mux selects the group c / (nRxChannels * rxInterleave), i.e. AFE58JD32: channels 0-63 or 64-127;
     * which channels share a slot is given by getRxInputTable().
     */
    uint32_t getRxInterleave() const { return rxInterleave; }
    /** The number of output data columns (RX slots * inputs per slot). */
    ChannelIdx getNRxOutputChannels() const { return static_cast<ChannelIdx>(nRxChannels * rxInterleave); }
    /** ADC clock frequency [Hz]: the rate of the raw (interleaved) RX slot stream. */
    float getAdcClockFrequency() const { return samplingFrequency * static_cast<float>(rxInterleave); }
    /** Physical channel -> (RX slot, raw row), see RxInputTable; required for rxInterleave > 1. */
    const RxInputTable &getRxInputTable() const { return rxInputTable; }
    RxInputLocation getRxInputLocation(ChannelIdx physicalChannel) const {
        return ::arrus::devices::getRxInputLocation(physicalChannel, nRxChannels, rxInterleave, rxInputTable);
    }

private:
    friend class Us4OEMDescriptorBuilder;
    uint32_t version{0};
    ChannelIdx nRxChannels;
    float minRxTime;
    float rxTimeEpsilon;
    float sequenceReprogrammingTime;
    float samplingFrequency;
    size_t ddrSize;
    size_t maxTransferSize;
    float nPeriodsResolution;
    bool master;
    arrus::ops::us4r::TxRxSequenceLimits txRxSequenceLimits;
    uint8_t nTimeouts{0};
    uint32_t sampleTxStart{0};
    uint32_t rxInterleave{1};
    RxInputTable rxInputTable;
};

class Us4OEMDescriptorBuilder {
public:

    explicit Us4OEMDescriptorBuilder(Us4OEMDescriptor descriptor): descriptor(std::move(descriptor)) {}

    Us4OEMDescriptorBuilder &setTxRxSequenceLimits(const ::arrus::ops::us4r::TxRxSequenceLimits &limits) {
        this->descriptor->txRxSequenceLimits = limits;
        return *this;
    }

    Us4OEMDescriptor build() {
        if(!descriptor.has_value()) {
            throw IllegalStateException("No parameters set for the new descriptor!");
        }
        auto result = descriptor.value();
        descriptor.reset();
        return result;
    }

private:
    std::optional<Us4OEMDescriptor> descriptor;

};

}

#endif //ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMLIMITS_H
