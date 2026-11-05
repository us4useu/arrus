#ifndef ARRUS_CORE_DEVICES_US4R_VALIDATORS_TXRXSEQUENCEVALIDATOR_H
#define ARRUS_CORE_DEVICES_US4R_VALIDATORS_TXRXSEQUENCEVALIDATOR_H

#include "arrus/core/api/ops/us4r/TxRxSequence.h"
#include "arrus/core/common/validation.h"

namespace arrus::devices {

/**
 * Validates TxRxSequence.
 *
 * TGC: exactly one of the following options is supported:
 * - Tgc set for each (non-NOP) Rx, not set in TxRxSequence,
 * - Tgc set in TxRxSequence, not set in any of Rx,
 * - Tgc not set for both TxRxSequence and Rx.
 */
class TxRxSequenceValidator : public Validator<ops::us4r::TxRxSequence> {
public:
    explicit TxRxSequenceValidator(const std::string &componentName) : Validator(componentName) {}

    void validate(const ops::us4r::TxRxSequence &sequence) override {
        validateTgc(sequence);
    }

private:
    void validateTgc(const ops::us4r::TxRxSequence &sequence) {
        const auto &ops = sequence.getOps();
        size_t nRxWithTgc = 0, nNonNopRx = 0, nNonNopRxWithTgc = 0;
        for(const auto &op: ops) {
            const auto &rx = op.getRx();
            const bool hasTgc = rx.getTgc().has_value();
            if(hasTgc) {
                ++nRxWithTgc;
            }
            if(!rx.isNOP()) {
                ++nNonNopRx;
                if(hasTgc) {
                    ++nNonNopRxWithTgc;
                }
            }
        }
        const bool sequenceTgc = sequence.getTgc().has_value();
        if(sequenceTgc) {
            ARRUS_VALIDATOR_EXPECT_TRUE_M(
                nRxWithTgc == 0,
                format("Sequence '{}': TGC curve is set for the TxRxSequence and for some of the Rxs; TGC curve "
                       "should be set either for the TxRxSequence or for each Rx.", sequence.getName()));
        } else if(nRxWithTgc > 0) {
            ARRUS_VALIDATOR_EXPECT_TRUE_M(
                nNonNopRxWithTgc == nNonNopRx,
                format("Sequence '{}': TGC curve is set only for some of the Rxs ({} of {}); TGC curve should be set "
                       "for each Rx (or for the whole TxRxSequence).", sequence.getName(), nNonNopRxWithTgc, nNonNopRx));
        }
        auto checkTgc = [this, &sequence](const ops::us4r::Tgc &tgc, const std::string &where) {
            ARRUS_VALIDATOR_EXPECT_TRUE_M(
                !tgc.empty(),
                format("Sequence '{}': TGC curve set for {} should not be empty.", sequence.getName(), where));
            if(tgc.hasTime()) {
                ARRUS_VALIDATOR_EXPECT_TRUE_M(
                    tgc.getTime().value().size() == tgc.getValues().size(),
                    format("Sequence '{}': TGC curve set for {}: the number of time points should be equal to the "
                           "number of values.", sequence.getName(), where));
            }
        };
        if(sequenceTgc) {
            checkTgc(sequence.getTgc().value(), "the TxRxSequence");
        }
        for(size_t i = 0; i < ops.size(); ++i) {
            const auto &tgc = ops.at(i).getRx().getTgc();
            if(tgc.has_value()) {
                checkTgc(tgc.value(), format("Rx #{}", i));
            }
        }
    }
};

}

#endif //ARRUS_CORE_DEVICES_US4R_VALIDATORS_TXRXSEQUENCEVALIDATOR_H
