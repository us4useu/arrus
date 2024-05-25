#ifndef ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMLIMITS_H
#define ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMLIMITS_H

#include "arrus/core/api/common/Interval.h"
#include "arrus/core/api/common/types.h"
#include "arrus/core/api/devices/us4r/RxSettings.h"
#include "arrus/core/api/ops/us4r/constraints/TxRxSequenceLimits.h"
#include <ius4oem.h>

namespace arrus::devices {

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

    Us4OEMDescriptor(ChannelIdx nRxChannels, float minRxTime, float rxTimeEpsilon, float sequenceReprogrammingTime,
                     float samplingFrequency, size_t ddrSize, size_t maxTransferSize, float nPeriodsResolution,
                     bool master, const ops::us4r::TxRxSequenceLimits &txRxSequenceLimits)
        : nRxChannels(nRxChannels), minRxTime(minRxTime), rxTimeEpsilon(rxTimeEpsilon),
          sequenceReprogrammingTime(sequenceReprogrammingTime), samplingFrequency(samplingFrequency), ddrSize(ddrSize),
          maxTransferSize(maxTransferSize), nPeriodsResolution(nPeriodsResolution),
          master(master), txRxSequenceLimits(txRxSequenceLimits) {}

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

private:
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
};

}

#endif //ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMLIMITS_H
