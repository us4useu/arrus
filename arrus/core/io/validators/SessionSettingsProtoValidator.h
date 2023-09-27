#ifndef ARRUS_CORE_IO_VALIDATORS_SESSIONSETTINGSPROTOVALIDATOR_H
#define ARRUS_CORE_IO_VALIDATORS_SESSIONSETTINGSPROTOVALIDATOR_H

#include "arrus/common/compiler.h"
#include "arrus/core/common/validation.h"

#include "arrus/core/io/validators/Us4RSettingsProtoValidator.h"
#include "arrus/core/io/validators/FileSettingsProtoValidator.h"

COMPILER_PUSH_DIAGNOSTIC_STATE
COMPILER_DISABLE_MSVC_WARNINGS(4127)
#include "io/proto/session/SessionSettings.pb.h"
COMPILER_POP_DIAGNOSTIC_STATE


namespace arrus::io {

class SessionSettingsProtoValidator: public Validator<std::unique_ptr<arrus::proto::SessionSettings>> {
public:
    explicit SessionSettingsProtoValidator(const std::string &name): Validator(name) {}

    void validate(
        const std::unique_ptr<arrus::proto::SessionSettings> &obj) override {
        expectTrue("session devices", obj->has_us4r() || obj->has_file(),
                   "exactly one of the following is required: {us4r, file}");

        if(hasErrors()) {
            return;
        }
        if(obj->has_us4r()) {
            Us4RSettingsProtoValidator validator("us4r");
            validator.validate(obj->us4r());
            copyErrorsFrom(validator);
        }
        else if(obj->has_file()) {
            FileSettingsProtoValidator validator("file");
            validator.validate(obj->file());
            copyErrorsFrom(validator);
        }
    }
};

}

#endif //ARRUS_CORE_IO_VALIDATORS_SESSIONSETTINGSPROTOVALIDATOR_H
