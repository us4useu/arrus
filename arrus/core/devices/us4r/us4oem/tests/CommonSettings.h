#ifndef ARRUS_CORE_DEVICES_US4R_US4OEM_TESTS_COMMONS_H
#define ARRUS_CORE_DEVICES_US4R_US4OEM_TESTS_COMMONS_H

#include "arrus/core/api/devices/us4r/Us4OEMSettings.h"
#include "arrus/core/api/ops/us4r/constraints/TxRxSequenceLimits.h"
#include "arrus/core/common/collections.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMDescriptor.h"

using namespace arrus;
using namespace arrus::devices;

struct TestUs4OEMSettings {
    std::vector<ChannelIdx> channelMapping{getRange<ChannelIdx>(0, 128)};
    std::optional<uint16> dtgcAttenuation{std::nullopt};
    uint16 pgaGain{30};
    uint16 lnaGain{24};
    RxSettings::TGCCurve tgcSamples{getRange<float>(30, 40, 0.5)};
    uint32 lpfCutoff{(int) 10e6};
    std::optional<uint16> activeTermination{50};
    bool isApplyCharacteristic{true};

    std::vector<std::string> invalidParameters;

    Us4OEMSettings getUs4OEMSettings() const {
        return Us4OEMSettings(
            channelMapping,
            RxSettings(dtgcAttenuation, pgaGain, lnaGain, tgcSamples, lpfCutoff,
                       activeTermination, isApplyCharacteristic));
    }

    friend std::ostream &
    operator<<(std::ostream &os, const TestUs4OEMSettings &settings) {
        os << "channelMapping: " << toString(settings.channelMapping)
           << " dtgcAttenuation: " << toString(settings.dtgcAttenuation)
           << " pgaGain: " << (int) settings.pgaGain
           << " lnaGain: " << (int) settings.lnaGain
           << " lpfCutoff: " << settings.lpfCutoff
           << " activeTermination: " << toString(settings.activeTermination)
           << " tgcSamples: " << toString(settings.tgcSamples);

        for(const auto &invalidParameter : settings.invalidParameters) {
            os << " invalidParameter: " << invalidParameter;
        }
        return os;
    }
};

Us4OEMDescriptor DEFAULT_DESCRIPTOR {
    32, // RX channels
    20e-6f,  // min. RX time,
    5e-6, // RX time epsilon,
    35e-6, // TX parameters reprogramming time,
    65e6, // Sampling frequency [Hz]
    1ull << 32u, // DDR memory size [B]
    1ull << (14+12), // Max transfer size [B]
    0.5f,  // number of TX periods resolution
    true,
    arrus::ops::us4r::TxRxSequenceLimits {
        arrus::ops::us4r::TxRxLimits {
            arrus::ops::us4r::TxLimits {
                Interval<float>{1e6, 60e6},  // Frequency
                Interval<float>{0.0f, 16.96e-6f}, // delay
                Interval<float>{0.0f, 32.0f/10e6}, // pulse length,
                Interval<Voltage>{5, 90}
            },
            arrus::ops::us4r::RxLimits {
                Interval<uint32>{64, 16384}
            },
            Interval<float>{35e-6f, 1.0f},  // PRI, == (the sequence reprogramming time, 1s)
        },
        Interval<uint32>{0, 16384} // sequence length
    }

};

struct TestTxRxParams {

    TestTxRxParams() {
        for (int i = 0; i < 32; ++i) {
            rxAperture[i] = true;
        }
    }

    BitMask txAperture = getNTimes(true, Us4OEMDescriptor::N_TX_CHANNELS);
    std::vector<float> txDelays = getNTimes(0.0f, Us4OEMDescriptor::N_TX_CHANNELS);
    ops::us4r::Pulse pulse{10.0e6f, 2.5f, true};
    BitMask rxAperture = getNTimes(false, Us4OEMDescriptor::N_ADDR_CHANNELS);
    uint32 decimationFactor = 1;
    float pri = 200e-6f;
    Interval<uint32> sampleRange{0, 4096};
    std::optional<BitstreamId> bitstreamId{std::nullopt};
    std::unordered_set<ChannelIdx> maskedChannelsTx = {};
    std::unordered_set<ChannelIdx> maskedChannelsRx = {};
    Tuple<ChannelIdx> rxPadding = {0, 0};
    float rxDelay = 0.0f;

    [[nodiscard]] arrus::devices::us4r::TxRxParameters get() const {
        return arrus::devices::us4r::TxRxParameters(
            txAperture, txDelays, pulse, rxAperture, sampleRange, decimationFactor, pri,
            rxPadding, rxDelay, bitstreamId, maskedChannelsTx, maskedChannelsRx);
    }
};

struct TestTxRxParamsSequence {
    std::vector<arrus::devices::us4r::TxRxParameters> txrx = {TestTxRxParams{}.get()};
    uint16 nRepeats = 1;
    std::optional<float> sri = std::nullopt;
    ops::us4r::TGCCurve tgcCurve = {};
    DeviceId txProbeId{arrus::devices::DeviceType::Probe, 0};
    DeviceId rxProbeId{arrus::devices::DeviceType::Probe, 0};

    [[nodiscard]] arrus::devices::us4r::TxRxParametersSequence get() const {
        return arrus::devices::us4r::TxRxParametersSequence {
            txrx, nRepeats, sri, tgcCurve, txProbeId, rxProbeId
        };
    }
};


#endif //ARRUS_CORE_DEVICES_US4R_US4OEM_TESTS_COMMONS_H
