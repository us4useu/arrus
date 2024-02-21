#ifndef ARRUS_CORE_DEVICES_US4R_VALIDATORS_PROBEADAPTERTXRXVALIDATOR_H
#define ARRUS_CORE_DEVICES_US4R_VALIDATORS_PROBEADAPTERTXRXVALIDATOR_H

#include "arrus/core/common/validation.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMImpl.h"

namespace arrus::devices {

class ProbeAdapterTxRxValidator : public Validator<us4r::TxRxParametersSequence> {
public:
    ProbeAdapterTxRxValidator(const std::string &componentName, ChannelIdx nChannels)
        : Validator(componentName), nChannels(nChannels) {}

    void validate(const us4r::TxRxParametersSequence &txRxs) override {
        auto const &op = txRxs.getFirstRxOp();
        ARRUS_REQUIRES_TRUE_IAE(op.has_value(), "There should be at least a single TX/RX with non-empty RX aperture");
        const auto &refOp = op.value();
        auto nSamples = refOp.getNumberOfSamples();
        size_t nActiveRxChannels =
            std::accumulate(std::begin(refOp.getRxAperture()), std::end(refOp.getRxAperture()), 0);
        nActiveRxChannels += refOp.getRxPadding().sum();
        for (size_t firing = 0; firing < txRxs.size(); ++firing) {
            const auto &op = txRxs.at(firing);
            auto firingStr = ::arrus::format("firing {}", firing);
            ARRUS_VALIDATOR_EXPECT_EQUAL_M(op.getRxAperture().size(), size_t(nChannels), firingStr);
            ARRUS_VALIDATOR_EXPECT_EQUAL_M(op.getTxAperture().size(), size_t(nChannels), firingStr);
            ARRUS_VALIDATOR_EXPECT_EQUAL_M(op.getTxDelays().size(), size_t(nChannels), firingStr);

            if (!op.isRxNOP()) {
                ARRUS_VALIDATOR_EXPECT_TRUE_M(op.getNumberOfSamples() == nSamples,
                                              "Each Rx should acquire the same number of samples.");
                size_t currActiveRxChannels = std::accumulate(std::begin(txRxs.at(firing).getRxAperture()),
                                                              std::end(txRxs.at(firing).getRxAperture()), 0);
                currActiveRxChannels += txRxs.at(firing).getRxPadding().sum();
                ARRUS_VALIDATOR_EXPECT_TRUE_M(currActiveRxChannels == nActiveRxChannels,
                                              "Each rx aperture should have the same size.");
            }
            if (hasErrors()) {
                return;
            }
        }
    }



private:
    ChannelIdx nChannels;
};
}// namespace arrus::devices

#endif// ARRUS_CORE_DEVICES_US4R_VALIDATORS_PROBEADAPTERTXRXVALIDATOR_H