#ifndef ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMDESCRIPTORFACTORY_H
#define ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMDESCRIPTORFACTORY_H

#include "Us4OEMDescriptor.h"
#include "arrus/core/devices/us4r/external/ius4oem/IUs4OEMFactory.h"
#include "arrus/core/api/ops/us4r/constraints/TxRxSequenceLimits.h"
#include "arrus/common/format.h"
#include <ius4oem.h>
#include <cstdint>
namespace arrus::devices {

class Us4OEMDescriptorFactory {
public:
    /**
     * us4OEM+ 64 (AFE58JD32) RX wiring: physical channel -> (RX slot, raw row).
     * Source: us4OEM+ 64 schematic (AFE58JD32 INP(2k+1)/INP(2k+2) <- RX_IN <- LVOUT; LVOUT[c] is muxed with
     * transducer channels c and c + 64); verified on the bench 2026-10-01 (single-element TX sweep, all 128 TX).
     * Converter k reads the odd input (raw row 0) and the even input (raw row 1). NOTE: the row assignment holds
     * for an odd acquisition start in ADC clocks (sampleTxStart = 35, start sample * 2); changing the parity of
     * sampleTxStart swaps the rows.
     * The AFE on RX slots 0-15 gets LVOUT 0-15 and 32-47; the AFE on slots 16-31 the same pattern + 16.
     */
    static RxInputTable createAfe58jd32RxInputTable() {
        // LVOUT of the odd/even input of converter k (k = 0..15) of the AFE on RX slots 0-15.
        constexpr uint8_t ODD[16] = {32, 34, 35, 33, 4, 7, 37, 38, 10, 9, 42, 41, 14, 12, 13, 15};
        constexpr uint8_t EVEN[16] = {0, 2, 3, 1, 6, 5, 39, 36, 8, 11, 40, 43, 46, 47, 45, 44};
        RxInputTable table(Us4OEMDescriptor::N_ADDR_CHANNELS);
        for (uint8_t afe = 0; afe < 2; ++afe) {
            for (uint8_t muxGroup = 0; muxGroup < 2; ++muxGroup) {
                const auto offset = static_cast<uint8_t>(afe * 16 + muxGroup * 64);
                for (uint8_t k = 0; k < 16; ++k) {
                    const auto slot = static_cast<uint8_t>(afe * 16 + k);
                    table.at(ODD[k] + offset) = std::pair<uint8_t, uint8_t>{slot, uint8_t{0}};
                    table.at(EVEN[k] + offset) = std::pair<uint8_t, uint8_t>{slot, uint8_t{1}};
                }
            }
        }
        return table;
    }

    static Us4OEMDescriptor getDescriptor(const IUs4OEMHandle &ius4oem, bool isMaster) {
        auto version = ius4oem->GetOemVersion();

        auto minFrequencyLegacy = ius4oem->GetMinTxFrequency();
        auto maxFrequencyLegacy = ius4oem->GetMaxTxFrequency();

        switch (version) {
        case 1:
            // Legacy us4OEM
            return Us4OEMDescriptor{
                version, // Us4OEM version
                32, // RX channels
                20e-6f,  // min. RX time,
                5e-6f, // RX time epsilon,
                35e-6f, // TX parameters reprogramming time,
                65e6f, // Sampling frequency [Hz]
                1ull << 32u, // DDR memory size [B]
                1ull << (14+12), // Max transfer size [B]
                0.5f,  // number of TX periods resolution
                isMaster,
                arrus::ops::us4r::TxRxSequenceLimits {
                    arrus::ops::us4r::TxRxLimits {
                        // amplitude 1 / rail 1
                        // UNAVAILABLE (voltages set to 0)
                        arrus::ops::us4r::TxLimits {
                            Interval<float>{minFrequencyLegacy, maxFrequencyLegacy},  // Frequency
                            Interval<float>{0.0f, 16.96e-6f}, // delay
                            Interval<float>{0.5f, (float)(32.0f)}, // pulse length in cycles,
                            Interval<Voltage>{0, 0} // UNAVAILABLE
                        },
                        // amplitude 2 / rail 0
                        arrus::ops::us4r::TxLimits {
                            Interval<float>{minFrequencyLegacy, maxFrequencyLegacy},  // Frequency
                            Interval<float>{0.0f, 16.96e-6f}, // delay
                            Interval<float>{0.5f, (float)(32.0f)}, // pulse length in cycles,
                            Interval<Voltage>{5, 90}
                        },
                        arrus::ops::us4r::RxLimits {
                            Interval<uint32>{64, 16384}
                        },
                        Interval<float>{35e-6f, 1.0f},  // PRI, == (the sequence reprogramming time, 1s)
                    },
                    Interval<uint32>{0, 16384}, // sequence length,
                    1024 // maximum number of different TX/RXs
                },
                0, // maximum number of TX timeouts
                229 // sample number TX start (i.e. TX delay = 0)
            };
        case 2:
            // us4OEM+ variant 0 AFE JD18
            return Us4OEMDescriptor{
                version, // us4OEM version
                32, // RX channels
                20e-6f,  // min. RX time,
                0e-6f, // RX time epsilon,
                7e-6f, // TX parameters reprogramming time,
                65e6f, // Sampling frequency [Hz]
                1ull << 32u, // DDR memory size [B]
                1ull << (14+12), // Max transfer size [B]
                0.5f,  // number of TX periods resolution
                isMaster,
                arrus::ops::us4r::TxRxSequenceLimits {
                    arrus::ops::us4r::TxRxLimits {
                        // amplitude 1 / rail 1
                        arrus::ops::us4r::TxLimits {
                            Interval<float>{100e3, 32.5e6},  // Frequency
                            Interval<float>{0.0f, 16.96e-6f}, // delay
                            Interval<float>{0.5f, (float)(32.0f)}, // pulse length in cycles,
                            Interval<Voltage>{5, 90}
                        },
                        // amplitude 2 / rail 0
                        arrus::ops::us4r::TxLimits {
                            Interval<float>{100e3, 32.5e6},  // Frequency
                            Interval<float>{0.0f, 16.96e-6f}, // delay
                            Interval<float>{0.5f, (float)(32.0f)}, // pulse length in cycles,
                            Interval<Voltage>{5, 90}
                        },
                        arrus::ops::us4r::RxLimits {
                            // The number of samples must be divisible by 64, therefore 65536-64 = 65472
                            Interval<uint32>{64, 65472}
                        },
                        Interval<float>{35e-6f, 1.0f},  // PRI, == (the sequence reprogramming time, 1s)
                    },
                    Interval<uint32>{0, 4096}, // sequence length
                    4096 // maximum number of different TX/RXs
                },
                4, // maximum number of timeouts
                35 // sample number TX start (i.e. TX delay = 0)
            };
        case 3:
            // us4OEM+ variant 0, AFE JD48
            return Us4OEMDescriptor{
                version, // us4OEM version
                32, // RX channels
                20e-6f,  // min. RX time,
                0e-6f, // RX time epsilon,
                35e-6f, // TX parameters reprogramming time,
                120e6f, // Sampling frequency [Hz]
                1ull << 32u, // DDR memory size [B]
                1ull << (14+12), // Max transfer size [B]
                0.5f,  // number of TX periods resolution
                isMaster,
                arrus::ops::us4r::TxRxSequenceLimits {
                    arrus::ops::us4r::TxRxLimits {
                        // amplitude 1 / rail 1
                        arrus::ops::us4r::TxLimits {
                            Interval<float>{100e3, 32.5e6},  // Frequency
                            Interval<float>{0.0f, 16.96e-6f}, // delay
                            Interval<float>{0.5f, (float)(32.0f)}, // pulse length in cycles,
                            Interval<Voltage>{5, 90}
                        },
                        // amplitude 2 / rail 0
                        arrus::ops::us4r::TxLimits {
                            Interval<float>{100e3, 32.5e6},  // Frequency
                            Interval<float>{0.0f, 16.96e-6f}, // delay
                            Interval<float>{0.5f, (float)(32.0f)}, // pulse length in cycles,
                            Interval<Voltage>{5, 90}
                        },
                        arrus::ops::us4r::RxLimits {
                            Interval<uint32>{64, 65472}
                        },
                        Interval<float>{35e-6f, 1.0f},  // PRI, == (the sequence reprogramming time, 1s)
                    },
                    Interval<uint32>{0, 4096} // sequence length
                },
                4, // maximum number of timeouts
                35 // sample number TX start (i.e. TX delay = 0)
            };
        case 4:
            // us4OEM+ 64, AFE JD32: 64 RX inputs; each of the 32 RX slots carries an odd/even input pair,
            // interleaved sample by sample at the 65 MHz ADC clock (32.5 MHz per input). The RX mux selects
            // channels 0-63 or 64-127 (c and c + 64 share an AFE input); which channels share an RX slot is given
            // by createAfe58jd32RxInputTable(). The odd/even phase is reset by the per-firing TX_TRIG.
            // TODO(JD32) bench: sampleTxStart (keep it odd, see createAfe58jd32RxInputTable); the raw row 1 inputs
            // are sampled 1 ADC clock (15.4 ns) after the raw row 0 inputs, not compensated yet.
            return Us4OEMDescriptor{
                version, // us4OEM version
                32, // RX slots (FPGA RX channel mapping size)
                20e-6f,  // min. RX time,
                0e-6f, // RX time epsilon,
                7e-6f, // TX parameters reprogramming time,
                32.5e6f, // Sampling frequency (per input) [Hz]
                1ull << 32u, // DDR memory size [B]
                1ull << (14+12), // Max transfer size [B]
                0.5f,  // number of TX periods resolution
                isMaster,
                arrus::ops::us4r::TxRxSequenceLimits {
                    arrus::ops::us4r::TxRxLimits {
                        // amplitude 1 / rail 1
                        arrus::ops::us4r::TxLimits {
                            Interval<float>{100e3, 32.5e6},  // Frequency
                            Interval<float>{0.0f, 16.96e-6f}, // delay
                            Interval<float>{0.5f, (float)(32.0f)}, // pulse length in cycles,
                            Interval<Voltage>{5, 90}
                        },
                        // amplitude 2 / rail 0
                        arrus::ops::us4r::TxLimits {
                            Interval<float>{100e3, 32.5e6},  // Frequency
                            Interval<float>{0.0f, 16.96e-6f}, // delay
                            Interval<float>{0.5f, (float)(32.0f)}, // pulse length in cycles,
                            Interval<Voltage>{5, 90}
                        },
                        arrus::ops::us4r::RxLimits {
                            // Per input; raw (ADC clock) samples = 2x, max. 65472.
                            Interval<uint32>{32, 32736}
                        },
                        Interval<float>{35e-6f, 1.0f},  // PRI, == (the sequence reprogramming time, 1s)
                    },
                    Interval<uint32>{0, 4096}, // sequence length
                    4096 // maximum number of different TX/RXs
                },
                4, // maximum number of timeouts
                35, // raw (ADC clock) sample number TX start (i.e. TX delay = 0)
                2, // RX interleave (inputs per RX slot)
                createAfe58jd32RxInputTable()
            };
        default:
            throw arrus::IllegalArgumentException(format("Unsupported us4OEM version: {}", version));
        }
    }

};

}

#endif//ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMDESCRIPTORFACTORY_H

