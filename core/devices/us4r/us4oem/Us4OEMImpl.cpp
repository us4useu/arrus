#include "Us4OEMImpl.h"

#include "arrus/core/common/collections.h"
#include "arrus/common/asserts.h"
#include "arrus/core/api/devices/us4r/Us4OEMSettings.h"
#include "arrus/core/common/hash.h"
#include "arrus/core/common/interpolate.h"

namespace arrus::devices {

Us4OEMImpl::Us4OEMImpl(DeviceId id, IUs4OEMHandle ius4oem,
                       const BitMask &activeChannelGroups,
                       std::vector<uint8_t> channelMapping)
    : Us4OEM(id), logger{getLoggerFactory()->getLogger()},
      ius4oem(std::move(ius4oem)),
      channelMapping(std::move(channelMapping)) {
    INIT_ARRUS_DEVICE_LOGGER(logger, id.toString());

    // This class stores reordered channels, according to IUs4OEM docs.
    static const size_t acgRemap[] = {0, 4, 8, 12,
                                      2, 6, 10, 14,
                                      1, 5, 9, 13,
                                      3, 7, 11, 15};
    auto acg = ::arrus::permute(activeChannelGroups, acgRemap);
    ARRUS_REQUIRES_TRUE(acg.size == activeChannelGroups.size(),
                        arrus::format(
                            "Invalid number of active channels mask elements; "
                            "the input has {}, expected: {}" acg.size(),
                            activeChannelGroups.size()));
    activeChannelGroups = ::arrus::toBitset<N_ACTIVE_CHANNEL_GROUPS>(acg);
}

Us4OEMImpl::~Us4OEMImpl() {
    try {
        logger->log(LogSeverity::DEBUG, arrus::format("Destroying handle"));
    } catch(const std::exception &e) {
        std::cerr <<
                  arrus::format("Exception while calling us4oem destructor: {}",
                                e.what())
                  << std::endl;
    }
}

void Us4OEMImpl::startTrigger() {

}

void Us4OEMImpl::stopTrigger() {

}

void Us4OEMImpl::setTxRxSequence(const std::vector<TxRxParameters> &seq,
                                 const ::arrus::ops::us4r::TGCCurve &tgc) {
    // maximum number of elements in sequence: 2048
    // TODO validate
    // tx aperture size: 128
    // tx delays - should contain exactly the number of active channels from the tx aperture
    // rx aperture size: 128, number of active elements <= 32
    // TGC: tgc is supported only for lna + pga = 54 (check us4r.m)
    // TODO initialize module: reset all parameters (turn off TGC, DTGC, ActiveTermination, etc.)
    // This probably should be implemented in IUs4OEMInitializer
    // TODO val: rx channel mapping
    // liczba unikalnych mapowan, ktore nalezy ustawic, nie powinna przekraczac 128
    // TODO number of samples must be divisible by 64
    // TODO rx time wynikajacy z liczby probek + epsilon nie moze przekroczyc 4000 us (ze skryptu matlabowego PK)
    // TODO TX frequency should be in range available for given us4oem (1-20MHz?)
    // TODO number of periods should be 0.5 or 1 or 1.5 or 2, or 2.5 .... maximum number of cycles: 32
    // TODO PRI should be in some range (lets say 100 - 1000 us) PRF: 1 Hz - 20kHz
    // maksymalna liczba profili opoznien: 1024

    // General sequence parameters.
    ius4oem->ClearScheduledReceive();
    ius4oem->SetNTriggers(seq.size());
    ius4oem->SetNumberOfFirings(seq.size());

    auto[rxMappings, rxApertures] = setRxMappings(seq);

    int firing = 0;
    size_t outputAddress = 0;
    for(auto const &op : seq) {
        logger->log(
            LogSeverity::TRACE,
            arrus::format("Setting tx/rx {}: {}", firing, op));
        // active channel groups already remapped in constructor
        ius4oem->SetActiveChannelGroup(activeChannelGroups, firing);

        // convert start rx sample and end rx sample to
        // rx channel mapping
        // Get number of unique channel mappings in this sequence
        // compute rxTime for given number of samples
        auto[startSample, endSample] = op.getRxSampleRange().asPair();
        size_t nSamples = endSample - startSample + 1;

        float rxTime = getRxTime(nSamples);

        size_t nBytes = nSamples * N_RX_CHANNELS * sizeof(OutputDType);
        auto rxMapId = rxMappings.find(firing)->second;

        // Tx
        ius4oem->SetTxAperture(
            ::arrus::toBitset<N_TX_CHANNELS>(op.getTxAperture()), firing);
        // Delays
        uint8 txChannel = 0;
        for(bool bit : op.getTxAperture()) {
            float txDelay = 0;
            if(bit) {
                txDelay = op.getTxDelays()[txChannel];
            }
            ius4oem->SetTxDelay(txChannel, txDelay, firing);
            ++txChannel;
        }
        ius4oem->SetTxFreqency(op.getTxPulse().getCenterFrequency(), firing);
        ius4oem->SetTxHalfPeriods(
            static_cast<uint8>(op.getTxPulse().getNPeriods() * 2), firing);
        ius4oem->SetTxInvert(op.getTxPulse().isInverse(), firing);
        ius4oem->SetTrigger(static_cast<short>(op.getPri() * 1e6), false,
                            firing);
        ius4oem->EnableSequencer();
        ius4oem->EnableTransmit();
        // Rx
        ius4oem->SetActiveChannelGroup(activeChannelGroups, firing);
        ius4oem->SetRxAperture(rxApertures[firing], firing);

        setTGC(tgc, firing);
        ius4oem->SetRxDelay(Us4OEMImpl::RX_DELAY, firing);
        ius4oem->SetRxTime(rxTime, firing);

        ius4oem->ScheduleReceive(firing, outputAddress, nSamples,
                                 TRIGGER_DELAY + startSample,
                                 op.getRxDecimationFactor(),
                                 rxMapId);
        outputAddress += nBytes;
        ++firing;
    }
}

std::pair<
    std::unordered_map<uint16, uint16>,
    std::vector<Us4OEMImpl::Us4rBitMask>>
Us4OEMImpl::setRxMappings(const std::vector<TxRxParameters> &seq) {
    // TODO this function should also return a channel mapping for each of the operation
    // so, for example, setting rx aperture to 0, 33, 66 should give rx mapping 0 -> 0, 33 -> 1, 66 -> 2
    // if e.g channels 33 and 66 are conflicting, should be 0-> 0, 33 -> 1, 66 -> ???
    // for all conflicting channels in the original rx aperture, set destination value to some "None", e.g. -1
    // TODO log information, which channels will be turned off
    std::unordered_map<uint16, uint16> result;
    std::unordered_map<std::vector<uint8>, uint16, ContainerHash<uint8>> rxMappings;

    // Rx apertures after taking into account possible conflicts in Rx channel
    // mapping.
    std::vector<Us4rBitMask> outputRxApertures;

    uint16 rxMapId = 0;
    uint16 opId = 0;
    for(const auto &op: seq) {
        // uint8 is required by us4r API.
        std::vector<uint8> mapping;
        std::unordered_set<uint8> channelsUsed;

        // Convert rx aperture + channel mapping -> rx channel mapping
        uint8 channel = 0;

        std::vector<uint8> conflictingChannels;
        std::bitset<N_ADDR_CHANNELS> outputRxAperture;

        for(const auto isOn : op.getRxAperture()) {
            if(isOn) {
                auto rxChannel = channel % N_RX_CHANNELS;
                // set rx channel mapping, even if the channel is conflicting -
                // this way we keep the expected shape of rx data as is
                // (with some "bad data" gaps).

                // STRATEGY: if there are conflicting rx channels, keep the
                // first one (with the lowest channel number), turn off all
                // the rest.
                mapping.push_back(rxChannel);
                // Turn off conflicting channels
                if(setContains(channelsUsed, channel)) {
                    conflictingChannels.push_back(channel);
                } else {
                    outputRxAperture[channel] = true;
                }

            }
            channel++;
        }
        outputRxApertures.push_back(outputRxAperture);

        auto mappingIt = rxMappings.find(mapping);
        if(mappingIt == std::end(rxMappings)) {
            rxMappings.emplace(mapping, rxMapId);
            result.emplace(opId, rxMapId);

            // Fill the rx channel mapping with 32 elements.
            for(auto ch: mapping) {
                channelsUsed.insert(ch);
            }
            // Move all the non active channels to the end of the mapping
            for(uint8 i = 0; i < N_RX_CHANNELS; ++i) {
                if(!setContains(channelsUsed, i)) {
                    mapping.push_back(i);
                }
            }

            // Set channel mapping
            ARRUS_REQUIRES_TRUE(mapping.size() == N_RX_CHANNELS,
                                arrus::format(
                                    "Invalid size of the RX "
                                    "channel mapping to set: {}",
                                    mapping.size()));
            ius4oem->SetRxChannelMapping(mapping, rxMapId);

            ++rxMapId;
        } else {
            result.emplace(opId, mappingIt->second);
        }
        ++opId;
    }
    return {result, outputRxApertures};
}

double Us4OEMImpl::getSamplingFrequency() {
    return Us4OEMImpl::SAMPLING_FREQUENCY;
}

float Us4OEMImpl::getRxTime(size_t nSamples) {
    return nSamples / Us4OEMImpl::SAMPLING_FREQUENCY
           + Us4OEMImpl::RX_TIME_EPSILON;
}

void Us4OEMImpl::setTGC(const ops::us4r::TGCCurve &tgc, uint16 firing) {
    if(tgc.empty()) {
        ius4oem->TGCDisable();
    } else {
        ius4oem->TGCEnable();

        static const std::vector<float> tgcChar =
            {14.000, 14.001, 14.002, 14.003, 14.024, 14.168, 14.480, 14.825,
             15.234, 15.770, 16.508, 17.382, 18.469, 19.796, 20.933, 21.862,
             22.891, 24.099, 25.543, 26.596, 27.651, 28.837, 30.265, 31.690,
             32.843, 34.045, 35.543, 37.184, 38.460, 39.680, 41.083, 42.740,
             44.269, 45.540, 46.936, 48.474, 49.895, 50.966, 52.083, 53.256,
             54};
        auto actualTGC = ::arrus::interpolate1d(
            tgcChar,
            ::arrus::getRange<float>(14, 54),
            tgc);

        std::vector<float> tgcNormalized(actualTGC.size());
        size_t i = 0;
        for(auto val : actualTGC) {
            tgcNormalized[i] = (val-14)/40;
        }
        ius4oem->TGCSetSamples(tgcNormalized, firing);
    }
}

}
