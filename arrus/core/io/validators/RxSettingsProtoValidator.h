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
        if(obj.dtgcAttenuation__case() == proto::RxSettings::kDtgcAttenuation) {
            expectDataType<uint16>("dtgc_attenuation", obj.dtgc_attenuation());
        }
        expectDataType<uint16>("pga_gain", obj.pga_gain());
        expectDataType<uint16>("lna_gain", obj.lna_gain());

        if(obj.activeTermination__case() == proto::RxSettings::kActiveTermination) {
            expectDataType<uint16>("active_termination", obj.active_termination());
        }
    }

};

}

#endif //ARRUS_CORE_IO_VALIDATORS_RXSETTINGSPROTOVALIDATOR_H
