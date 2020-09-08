#ifndef ARRUS_CORE_IO_VALIDATORS_PROBEMODELPROTOVALIDATOR_H
#define ARRUS_CORE_IO_VALIDATORS_PROBEMODELPROTOVALIDATOR_H

#include "arrus/common/compiler.h"
#include "arrus/core/common/validation.h"

COMPILER_PUSH_DIAGNOSTIC_STATE
COMPILER_DISABLE_MSVC_WARNINGS(4127)

#include "io/proto/devices/probe/ProbeModel.pb.h"

COMPILER_POP_DIAGNOSTIC_STATE

namespace arrus::io {

class ProbeModelProtoValidator : public Validator<arrus::proto::ProbeModel> {
public:
    explicit ProbeModelProtoValidator(const std::string &componentName)
        : Validator(componentName) {}

    void validate(const arrus::proto::ProbeModel &obj) override {
        // Data type
        expectAllDataType<ChannelIdx>("n_elements", obj.n_elements());
    }
};

}


#endif //ARRUS_CORE_IO_VALIDATORS_PROBEMODELPROTOVALIDATOR_H
