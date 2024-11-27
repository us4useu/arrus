#ifndef ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMLIMITS_H
#define ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMLIMITS_H

#include "arrus/core/api/common/Interval.h"
#include "arrus/core/api/common/types.h"
#include "arrus/core/api/devices/us4r/RxSettings.h"
#include "arrus/core/api/ops/us4r/constraints/TxRxSequenceLimits.h"
#include <ius4oem.h>

#include <utility>

namespace arrus::devices {

class Us4OEMDescriptorBuilder;

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
        bool master, ops::us4r::TxRxSequenceLimits txRxSequenceLimits, uint8_t nTimeouts, uint32_t sampleTxStart)
        : version(version), nRxChannels(nRxChannels), minRxTime(minRxTime), rxTimeEpsilon(rxTimeEpsilon),
          sequenceReprogrammingTime(sequenceReprogrammingTime), samplingFrequency(samplingFrequency), ddrSize(ddrSize),
          maxTransferSize(maxTransferSize), nPeriodsResolution(nPeriodsResolution),
          master(master), txRxSequenceLimits(std::move(txRxSequenceLimits)), nTimeouts(nTimeouts),
          sampleTxStart(sampleTxStart) {}

    bool isUs4OEMLegacy() {return version == US4OEM_LEGACY_REVISION; }
    bool isUs4OEMPlus() {return version == US4OEM_PLUS_REVISION; }
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
