#ifndef ARRUS_CORE_DEVICES_US4R_US4OEM_TESTS_COMMONS_H
#define ARRUS_CORE_DEVICES_US4R_US4OEM_TESTS_COMMONS_H

#include "arrus/core/common/collections.h"
#include "arrus/core/api/devices/us4r/Us4OEMSettings.h"

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


#endif //ARRUS_CORE_DEVICES_US4R_US4OEM_TESTS_COMMONS_H
