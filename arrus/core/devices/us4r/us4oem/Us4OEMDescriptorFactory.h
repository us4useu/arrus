#ifndef ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMDESCRIPTORFACTORY_H
#define ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMDESCRIPTORFACTORY_H

#include "Us4OEMDescriptor.h"
#include "arrus/core/devices/us4r/external/ius4oem/IUs4OEMFactory.h"
#include "arrus/core/api/ops/us4r/constraints/TxRxSequenceLimits.h"
#include "arrus/common/asserts.h"
#include "arrus/common/format.h"
#include "arrus/core/api/common/exceptions.h"
#include <ius4oem.h>
#include <cstdint>
#include <cstdlib>
#include <string>
namespace arrus::devices {

class Us4OEMDescriptorFactory {
public:

    /**
     * The largest single transfer from a board's DDR4 memory to the host.
     *
     * 64 MiB is the PCIe DMA's limit. Over Ethernet a transfer is instead one frame egressed by the
     * board's bridge, and the zero-copy RDMA receiver requires every frame to land at a uniform
     * pitch inside one host buffer -- which only holds when a buffer element is exactly ONE transfer
     * per board: two transfers per element put the second element's first frame an element stride
     * away, not a frame away, and the receiver refuses ("every destination must sit at a uniform
     * pitch"). A 192-transmit STA element is about 105 MiB per board, so the limit has to be raised
     * there. ARRUS_MAX_TRANSFER_SIZE, in bytes, does that; unset, nothing changes.
     */
    static size_t getMaxTransferSize() {
        static const size_t value = readMaxTransferSize();
        return value;
    }

    static size_t readMaxTransferSize() {
        constexpr size_t DEFAULT_MAX_TRANSFER_SIZE = 1ull << (14 + 12);// 64 MiB, the PCIe DMA limit
        const char *env = std::getenv("ARRUS_MAX_TRANSFER_SIZE");
        if (env == nullptr || *env == '\0') { return DEFAULT_MAX_TRANSFER_SIZE; }
        size_t value = 0;
        try {
            value = std::stoull(env);
        } catch (const std::exception &) {
            throw IllegalArgumentException(format("ARRUS_MAX_TRANSFER_SIZE must be a number of bytes, got: {}", env));
        }
        ARRUS_REQUIRES_TRUE_E(value > 0, IllegalArgumentException("ARRUS_MAX_TRANSFER_SIZE must be greater than 0"));
        return value;
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
                getMaxTransferSize(), // Max transfer size [B]
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
                getMaxTransferSize(), // Max transfer size [B]
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
                getMaxTransferSize(), // Max transfer size [B]
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
        default:
            throw arrus::IllegalArgumentException(format("Unsupported us4OEM version: {}", version));
        }
    }

};

}

#endif//ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMDESCRIPTORFACTORY_H

