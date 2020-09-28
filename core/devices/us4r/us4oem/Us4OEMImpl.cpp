#include "Us4OEMImpl.h"

#include <cmath>

#include "arrus/core/common/collections.h"
#include "arrus/common/asserts.h"
#include "arrus/core/api/devices/us4r/Us4OEMSettings.h"
#include "arrus/core/common/hash.h"
#include "arrus/core/common/interpolate.h"
#include "arrus/core/common/validation.h"
#include "arrus/core/devices/us4r/FrameChannelMappingImpl.h"

namespace arrus::devices {

Us4OEMImpl::Us4OEMImpl(DeviceId id, IUs4OEMHandle ius4oem,
                       const BitMask &activeChannelGroups,
                       std::vector<uint8_t> channelMapping,
                       uint16 pgaGain, uint16 lnaGain)
    : Us4OEM(id), logger{getLoggerFactory()->getLogger()},
      ius4oem(std::move(ius4oem)),
      channelMapping(std::move(channelMapping)),
      pgaGain(pgaGain), lnaGain(lnaGain) {

    INIT_ARRUS_DEVICE_LOGGER(logger, id.toString());

    // This class stores reordered active groups of channels,
    // as presented in the IUs4OEM docs.
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

class Us4OEMTxRxValidator : public Validator<TxRxParamsSequence> {
public:
    using Validator<TxRxParamsSequence>::Validator;

    void validate(const TxRxParamsSequence &txRxs) override {
        // Validation according to us4oem technote
        ARRUS_VALIDATOR_EXPECT_IN_RANGE(txRxs.size(), size_t(1), size_t(2048));
        for(size_t firing = 0; firing < txRxs.size(); ++firing) {
            const auto &op = txRxs[firing];
            if(!op.isNOP()) {
                auto firingStr = ::arrus::format("firing {}", firing);

                // Tx
                ARRUS_VALIDATOR_EXPECT_EQUAL_M(
                    op.getTxAperture().size(), size_t(128), firingStr);
                ARRUS_VALIDATOR_EXPECT_EQUAL_M(
                    op.getTxDelays().size(), size_t(128), firingStr);
                ARRUS_VALIDATOR_EXPECT_ALL_IN_RANGE_VM(
                    op.getTxDelays(), 0.0f, 19.96e-6f, firingStr);

                // Tx - pulse
                ARRUS_VALIDATOR_EXPECT_IN_RANGE_M(
                    op.getTxPulse().getCenterFrequency(), 1e6f, 20e6f, firingStr);
                ARRUS_VALIDATOR_EXPECT_IN_RANGE_M(
                    op.getTxPulse().getNPeriods(), 0.0f, 32.0f, firingStr);
                float ignore = 0.0f;
                float fractional = std::modf(op.getTxPulse().getNPeriods(), &ignore);
                ARRUS_VALIDATOR_EXPECT_TRUE_M(
                    (fractional == 0.0f || fractional == 0.5f),
                    (firingStr + ", n periods"));

                // Rx
                ARRUS_VALIDATOR_EXPECT_EQUAL_M(
                    op.getRxAperture().size(), size_t(128), firingStr);
                size_t numberOfActiveRxChannels = std::accumulate(
                    std::begin(op.getRxAperture()), std::end(op.getRxAperture()),
                    false);
                ARRUS_VALIDATOR_EXPECT_IN_RANGE_M(
                    numberOfActiveRxChannels, size_t(0), size_t(32), firingStr);
                uint32 numberOfSamples = op.getNumberOfSamples();
                ARRUS_VALIDATOR_EXPECT_IN_RANGE_M(
                // should be enough for condition rxTime < 4000 [us]
                    numberOfSamples, 64, 16384, firingStr);
                ARRUS_VALIDATOR_EXPECT_DIVISIBLE_M(
                    numberOfSamples, 64, firingStr);

                ARRUS_VALIDATOR_EXPECT_IN_RANGE_M(
                    op.getRxDecimationFactor(), 0, 5, firingStr);
                // TODO should be the same for all operations
                // TODO start sample should the same for all operations in a sequence
                ARRUS_VALIDATOR_EXPECT_IN_RANGE_M(
                    op.getPri(), 50e-6f, 1.0f, firingStr);
            }
        }
    }
};

std::pair<float, float> getTgcMinMax(uint16 pgaGain, uint16 lnaGain) {
    float max = float(pgaGain) + float(lnaGain);
    return std::make_pair(max - 40, max);
}

class TGCCurveValidator : public Validator<::arrus::ops::us4r::TGCCurve> {
public:
    TGCCurveValidator(const std::string &componentName, uint16 pgaGain, uint16 lnaGain)
        : Validator(componentName), pgaGain(pgaGain), lnaGain(lnaGain) {}

    void validate(const ops::us4r::TGCCurve &tgcCurve) override {
        if(pgaGain != 30 || lnaGain != 24) {
            ARRUS_VALIDATOR_EXPECT_TRUE_M(
                tgcCurve.empty(),
                "Currently TGC is supported only for "
                "PGA gain 30 dB, LNA gain 24 dB");
        } else {
            ARRUS_VALIDATOR_EXPECT_IN_RANGE(
                tgcCurve.size(), size_t(0), size_t(1022));
            auto[min, max] = getTgcMinMax(pgaGain, lnaGain);
            ARRUS_VALIDATOR_EXPECT_ALL_IN_RANGE_V(tgcCurve, min, max);
        }
    }

private:
    uint16 pgaGain, lnaGain;
};

FrameChannelMapping::Handle
Us4OEMImpl::setTxRxSequence(const TxRxParamsSequence &seq,
                            const ::arrus::ops::us4r::TGCCurve &tgc) {
    // TODO initialize module: reset all parameters (turn off TGC, DTGC, ActiveTermination, etc.)
    // This probably should be implemented in IUs4OEMInitializer

    // Validate input sequence and parameters.
    std::string deviceIdStr = getDeviceId().toString();
    Us4OEMTxRxValidator seqValidator(format("{} tx rx sequence", deviceIdStr));
    seqValidator.validate(seq);
    seqValidator.throwOnErrors();

    TGCCurveValidator tgcValidator(format("{} tgc samples", deviceIdStr), pgaGain, lnaGain);
    tgcValidator.validate(tgc);
    tgcValidator.throwOnErrors();

    // General sequence parameters.
    ius4oem->ClearScheduledReceive();
    ius4oem->SetNTriggers(seq.size());
    ius4oem->SetNumberOfFirings(seq.size());

    auto[rxMappings, rxApertures, fcm] = setRxMappings(seq);

    // helper data
    const std::bitset<N_ADDR_CHANNELS> emptyAperture;
    const std::bitset<N_ACTIVE_CHANNEL_GROUPS> emptyChannelGroups;
    // us4oem rxdma output address
    size_t outputAddress = 0;

    for(size_t firing = 0; firing < seq.size(); ++firing) {
        auto const &op = seq[firing];
        if(op.isNOP()) {
            logger->log(LogSeverity::TRACE,
                        format("Setting tx/rx {}: NOP {}", firing, op));
        } else {
            logger->log(LogSeverity::TRACE,
                        arrus::format("Setting tx/rx {}: {}", firing, op));
        }
        auto[startSample, endSample] = op.getRxSampleRange().asPair();
        size_t nSamples = endSample - startSample + 1;
        float rxTime = getRxTime(nSamples);
        size_t nBytes = nSamples * N_RX_CHANNELS * sizeof(OutputDType);
        auto rxMapId = rxMappings.find(firing)->second;

        if(op.isNOP()) {
            ius4oem->SetActiveChannelGroup(emptyChannelGroups, firing);
            ius4oem->SetTxAperture(emptyAperture, firing);
            ius4oem->SetRxAperture(emptyAperture, firing);
        } else {
            // active channel groups already remapped in constructor
            ius4oem->SetActiveChannelGroup(activeChannelGroups, firing);
            ius4oem->SetTxAperture(
                ::arrus::toBitset<N_TX_CHANNELS>(op.getTxAperture()), firing);
            ius4oem->SetRxAperture(rxApertures[firing], firing);
        }

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

        setTGC(tgc, firing);
        ius4oem->SetRxDelay(Us4OEMImpl::RX_DELAY, firing);
        ius4oem->SetRxTime(rxTime, firing);

        ARRUS_REQUIRES_AT_MOST(outputAddress + nBytes, (1ull << 32u),
                               ::arrus::format(
                                   "Total data size cannot exceed 4GiB (device {})",
                                   getDeviceId().toString()));
        if(op.isRxNOP()) {
            // Fake data acquisition
            ius4oem->ScheduleReceive(firing, outputAddress, 64,
                                     0, 0, rxMapId);
            // Do not move outputAddress pointer, fake data should be overwritten
            // by the next non-rx-nop (or ignored, if this is the last operation).
        } else {
            ius4oem->ScheduleReceive(firing, outputAddress, nSamples,
                                     SAMPLE_DELAY + startSample,
                                     op.getRxDecimationFactor() + 1,
                                     rxMapId);
            outputAddress += nBytes;
        }
    }

    ius4oem->EnableSequencer();
    ius4oem->EnableTransmit();
    return std::move(fcm);
}

std::tuple<
    std::unordered_map<uint16, uint16>,
    std::vector<Us4OEMImpl::Us4rBitMask>,
    FrameChannelMapping::Handle>
Us4OEMImpl::setRxMappings(const std::vector<TxRxParameters> &seq) {
    // a map: op ordinal number -> rx map id
    std::unordered_map<uint16, uint16> result;
    std::unordered_map<std::vector<uint8>, uint16, ContainerHash<uint8>> rxMappings;

    // FC mapping
    size_t numberOfOutputFrames = getNumberOfNoRxNOPs(seq);
    FrameChannelMappingBuilder fcmBuilder(numberOfOutputFrames, N_RX_CHANNELS);

    // Rx apertures after taking into account possible conflicts in Rx channel
    // mapping.
    std::vector<Us4rBitMask> outputRxApertures;

    uint16 rxMapId = 0;
    uint16 opId = 0;
    uint16 noRxNopId = 0;
    for(const auto &op: seq) {
        // Considering rx nops: rx channel mapping will be equal [0, 1,.. 31].
        // uint8 is required by us4r API.
        std::vector<uint8> mapping;
        std::unordered_set<uint8> channelsUsed;

        // Convert rx aperture + channel mapping -> rx channel mapping

        std::vector<uint8> conflictingChannels;
        std::bitset<N_ADDR_CHANNELS> outputRxAperture;

        uint8 channel = 0;
        uint8 onChannel = 0;
        bool isRxNop = true; // Will be false, if at least one channel is on (no rxnop).
        for(const auto isOn : op.getRxAperture()) {
            if(isOn) {
                isRxNop = false;
                auto rxChannel = channelMapping[channel];
                rxChannel = rxChannel % N_RX_CHANNELS;
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
                    ARRUS_REQUIRES_TRUE_E(
                        onChannel < N_RX_CHANNELS,
                        ArrusException("Up to 32 active rx channels can be set."));
                    fcmBuilder.setChannelMapping(noRxNopId, onChannel, noRxNopId, onChannel);
                } else {
                    outputRxAperture[channel] = true;
                    fcmBuilder.setChannelMapping(noRxNopId, onChannel, noRxNopId, FrameChannelMapping::UNAVAILABLE);
                }
                ++onChannel;
            }
            ++channel;
        }
        outputRxApertures.push_back(outputRxAperture);

        auto mappingIt = rxMappings.find(mapping);
        if(mappingIt == std::end(rxMappings)) {
            // Create new Rx channel mapping.
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
            ARRUS_REQUIRES_TRUE(
                rxMapId < 128,
                arrus::format("128 different rx mappings can be loaded only"
                              ", deviceId: {}.", getDeviceId().toString()));
            ius4oem->SetRxChannelMapping(mapping, rxMapId);
            ++rxMapId;
        } else {
            // Use the existing one.
            result.emplace(opId, mappingIt->second);
        }
        ++opId;
        if(!isRxNop) {
            ++noRxNopId;
        }
    }
    return {result, outputRxApertures, fcmBuilder.build()};
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
            tgcNormalized[i] = (val - 14) / 40;
        }
        ius4oem->TGCSetSamples(tgcNormalized, firing);
    }
}

}
