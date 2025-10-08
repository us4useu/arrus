#ifndef ARRUS_CORE_DEVICES_US4R_VALIDATORS_PROBETXRXVALIDATOR_H
#define ARRUS_CORE_DEVICES_US4R_VALIDATORS_PROBETXRXVALIDATOR_H

#include <utility>

#include "arrus/core/common/validation.h"
#include "arrus/core/devices/TxRxParameters.h"

namespace arrus::devices {

class ProbeTxRxValidator : public Validator<us4r::TxRxParametersSequence> {
public:
    ProbeTxRxValidator(const std::string &componentName, ProbeModel probeTx, ProbeModel probeRx)
        : Validator(componentName), probeTx(std::move(probeTx)), probeRx(std::move(probeRx)) {}

    void validate(const us4r::TxRxParametersSequence &sequence) override {

        auto nChannelsTx = probeTx.getNumberOfElements().product();
        auto nChannelsRx = probeRx.getNumberOfElements().product();
        auto &txFrequencyRange = probeTx.getTxFrequencyRange();

        for (size_t firing = 0; firing < sequence.size(); ++firing) {
            const auto &op = sequence.at(firing);
            auto firingStr = format(" (firing {})", firing);
            ARRUS_VALIDATOR_EXPECT_EQUAL_M(op.getTxAperture().size(), nChannelsTx, firingStr);
            ARRUS_VALIDATOR_EXPECT_EQUAL_M(op.getTxDelays().size(), nChannelsTx, firingStr);
            const auto pulse = ::arrus::ops::us4r::Pulse::fromWaveform(op.getTxWaveform());
            if(pulse.has_value()) {
                ARRUS_VALIDATOR_EXPECT_IN_RANGE_M(pulse.value().getCenterFrequency(), txFrequencyRange.start(),
                                                  txFrequencyRange.end(), firingStr);
            }
            // TODO else what?

            ARRUS_VALIDATOR_EXPECT_EQUAL_M(op.getRxAperture().size(), nChannelsRx, firingStr);
        }
    }
private:
    ProbeModel probeTx, probeRx;
};

}// namespace arrus::devices

#endif// ARRUS_CORE_DEVICES_US4R_VALIDATORS_PROBETXRXVALIDATOR_H