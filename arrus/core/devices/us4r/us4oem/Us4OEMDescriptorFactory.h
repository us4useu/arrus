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

    static Us4OEMDescriptor getDescriptor(const IUs4OEMHandle &ius4oem, bool isMaster) {
        auto version = ius4oem->GetOemVersion();
        auto minFrequency = ius4oem->GetMinTxFrequency();
        auto maxFrequency = ius4oem->GetMaxTxFrequency();
        switch (version) {
        case 1:
            // Legacy us4OEM
            return Us4OEMDescriptor{
                version, // Us4OEM version
                32, // RX channels
                20e-6f,  // min. RX time,
                5e-6, // RX time epsilon,
                35e-6, // TX parameters reprogramming time,
                65e6, // Sampling frequency [Hz]
                1ull << 32u, // DDR memory size [B]
                1ull << (14+12), // Max transfer size [B]
                0.5f,  // number of TX periods resolution
                isMaster,
                arrus::ops::us4r::TxRxSequenceLimits {
                    arrus::ops::us4r::TxRxLimits {
                        arrus::ops::us4r::TxLimits {
                            Interval<float>{minFrequency, maxFrequency},  // Frequency
                            Interval<float>{0.0f, 16.96e-6f}, // delay
                            Interval<float>{0.0f, 32.0f/10e6}, // pulse length,
                            Interval<Voltage>{5, 90}
                        },
                        arrus::ops::us4r::RxLimits {
                            Interval<uint32>{64, 16384}
                        },
                        Interval<float>{35e-6f, 1.0f},  // PRI, == (the sequence reprogramming time, 1s)
                    },
                    Interval<uint32>{0, 16383} // sequence length
                },
                0 // maximum number of TX timeouts
            };
        case 2:
            // us4OEM+ variant 0
            return Us4OEMDescriptor{
                version, // us4OEM version
                32, // RX channels
                20e-6f,  // min. RX time,
                5e-6, // RX time epsilon,
                35e-6, // TX parameters reprogramming time,
                65e6, // Sampling frequency [Hz]
                1ull << 32u, // DDR memory size [B]
                1ull << (14+12), // Max transfer size [B]
                0.5f,  // number of TX periods resolution
                isMaster,
                arrus::ops::us4r::TxRxSequenceLimits {
                    arrus::ops::us4r::TxRxLimits {
                        arrus::ops::us4r::TxLimits {
                            Interval<float>{minFrequency, maxFrequency},  // Frequency
                            Interval<float>{0.0f, 16.96e-6f}, // delay
                            Interval<float>{0.0f, 32.0f/10e6}, // pulse length,
                            Interval<Voltage>{5, 90}
                        },
                        arrus::ops::us4r::RxLimits {
                            Interval<uint32>{64, 16384}
                        },
                        Interval<float>{35e-6f, 1.0f},  // PRI, == (the sequence reprogramming time, 1s)
                    },
                    Interval<uint32>{0, 2047} // sequence length
                },
                4 // maximum number of timeouts
            };
        default:
            throw arrus::IllegalArgumentException(format("Unsupported us4OEM version: {}", version));
        }
    }

};

}

#endif//ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMDESCRIPTORFACTORY_H

