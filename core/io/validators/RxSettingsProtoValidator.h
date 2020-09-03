#ifndef ARRUS_CORE_IO_VALIDATORS_RXSETTINGSPROTOVALIDATOR_H
#define ARRUS_CORE_IO_VALIDATORS_RXSETTINGSPROTOVALIDATOR_H

#include "arrus/common/compiler.h"
#include "arrus/core/common/validation.h"

COMPILER_PUSH_DIAGNOSTIC_STATE
COMPILER_DISABLE_MSVC_WARNINGS(4127)

#include "io/proto/devices/us4r/RxSettings.pb.h"

COMPILER_POP_DIAGNOSTIC_STATE

namespace arrus::io {

class RxSettingsProtoValidator : public Validator<arrus::proto::RxSettings> {
    using Validator::Validator;

public:
    void validate(const arrus::proto::RxSettings &obj) override {
        bool hasTgcLinearFunction = obj.has_tgc_curve_linear();
        bool hasTgcSamples = !obj.tgc_samples().empty();
        expectTrue("tgc curve",
                   !(hasTgcLinearFunction && hasTgcSamples),
                   "Duplicated definition of tgc curve");
    }

};

}

#endif //ARRUS_CORE_IO_VALIDATORS_RXSETTINGSPROTOVALIDATOR_H
