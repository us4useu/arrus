#ifndef ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMTXRXVALIDATOR_H
#define ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMTXRXVALIDATOR_H

#include <chrono>
#include <cmath>
#include <utility>
#include "arrus/core/devices/TxRxParameters.h"
#include "arrus/core/common/validation.h"
#include "Us4OEMDescriptor.h"

namespace arrus::devices {

 class Us4OEMTxRxValidator : public Validator<::arrus::devices::us4r::TxRxParametersSequence> {
 public:
    Us4OEMTxRxValidator(const std::string &componentName, Us4OEMDescriptor descriptor, BitstreamId nBitstreams)
        : Validator(componentName), descriptor(std::move(descriptor)), nBitstreams(nBitstreams) {}

    void validate(const ::arrus::devices::us4r::TxRxParametersSequence &txRxs) {
        // TODO validate channel masking (if the channel numbers are in the appropriate range)
        // TODO validate sequence size
        // Validation according to us4oem technote
        const auto decimationFactor = txRxs.at(0).getRxDecimationFactor();
        const auto startSample = txRxs.at(0).getRxSampleRange().start();
        const auto& sequenceLimits = descriptor.getTxRxSequenceLimits();
        const auto& txRxLimits = sequenceLimits.getTxRx();
        const auto& txLimits = txRxLimits.getTx();
        const auto& rxLimits = txRxLimits.getRx();

        for (size_t firing = 0; firing < txRxs.size(); ++firing) {
            const auto &op = txRxs.at(firing);
            if (!op.isNOP()) {
                auto firingStr = ::arrus::format(" (firing {})", firing);
                // Tx
                ARRUS_VALIDATOR_EXPECT_EQUAL_M(op.getTxAperture().size(), size_t(descriptor.getNTxChannels()), firingStr);
                ARRUS_VALIDATOR_EXPECT_EQUAL_M(op.getTxDelays().size(), size_t(descriptor.getNTxChannels()), firingStr);
                ARRUS_VALIDATOR_EXPECT_ALL_IN_INTERVAL_VM(op.getTxDelays(), txLimits.getDelay(), firingStr);
                ARRUS_VALIDATOR_EXPECT_IN_RANGE_M(op.getTxPulse().getAmplitudeLevel(),
                                                  static_cast<ops::us4r::Pulse::AmplitudeLevel>(0),
                                                  static_cast<ops::us4r::Pulse::AmplitudeLevel>(1),
                                                  firingStr);

                // Tx - pulse
                ARRUS_VALIDATOR_EXPECT_IN_INTERVAL_M(op.getTxPulse().getCenterFrequency(), txLimits.getFrequency(), firingStr);
                float pulseLength = op.getTxPulse().getNPeriods()/op.getTxPulse().getCenterFrequency();
                ARRUS_VALIDATOR_EXPECT_IN_INTERVAL_M(pulseLength, txLimits.getPulseLength(), firingStr);
                float ignore = 0.0f;
                float fractional = std::modf(op.getTxPulse().getNPeriods(), &ignore);
                ARRUS_VALIDATOR_EXPECT_TRUE_M((fractional == 0.0f || fractional == 0.5f), (firingStr + ", n periods"));
                // Rx
                ARRUS_VALIDATOR_EXPECT_EQUAL_M(op.getRxAperture().size(), size_t(descriptor.getNAddressableRxChannels()), firingStr);
                size_t numberOfActiveRxChannels =
                    std::accumulate(std::begin(op.getRxAperture()), std::end(op.getRxAperture()), 0);
                ARRUS_VALIDATOR_EXPECT_IN_RANGE_M(numberOfActiveRxChannels, size_t(0), size_t(descriptor.getNRxChannels()), firingStr);
                uint32 numberOfSamples = op.getNumberOfSamples();
                ARRUS_VALIDATOR_EXPECT_IN_INTERVAL_M(numberOfSamples, rxLimits.getNSamples(), firingStr);
                ARRUS_VALIDATOR_EXPECT_DIVISIBLE_M(numberOfSamples, 64u, firingStr);
                ARRUS_VALIDATOR_EXPECT_IN_RANGE_M(op.getRxDecimationFactor(), 0, 10, firingStr);
                ARRUS_VALIDATOR_EXPECT_IN_INTERVAL_M(op.getPri(), txRxLimits.getPri(), firingStr);
                ARRUS_VALIDATOR_EXPECT_TRUE_M(op.getRxDecimationFactor() == decimationFactor,
                                              "Decimation factor should be the same for all operations." + firingStr);
                ARRUS_VALIDATOR_EXPECT_TRUE_M(op.getRxSampleRange().start() == startSample,
                                              "Start sample should be the same for all operations." + firingStr);
                ARRUS_VALIDATOR_EXPECT_TRUE_M((op.getRxPadding() == ::arrus::Tuple<ChannelIdx>{0, 0}),
                                              ("Rx padding is not allowed for us4oems. " + firingStr));
                // Channel masking
                ARRUS_VALIDATOR_EXPECT_ALL_IN_RANGE_IM(
                    op.getMaskedChannelsTx(),
                    ChannelIdx(0), ChannelIdx(Us4OEMDescriptor::N_TX_CHANNELS-1),
                    firingStr
                    );
                ARRUS_VALIDATOR_EXPECT_ALL_IN_RANGE_IM(
                    op.getMaskedChannelsRx(),
                    ChannelIdx(0), ChannelIdx(Us4OEMDescriptor::N_ADDR_CHANNELS-1),
                    firingStr
                );
            }
            if (op.getBitstreamId().has_value() && descriptor.isMaster()) {
                ARRUS_REQUIRES_TRUE(op.getBitstreamId().value() < nBitstreams,
                                    "Bitstream id should not exceed " + std::to_string(nBitstreams));

            }
        }
    }

private:
    Us4OEMDescriptor descriptor;
    BitstreamId nBitstreams;
};
}

#endif //ARRUS_ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMTXRXVALIDATOR_H
