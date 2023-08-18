#ifndef ARRUS_CORE_IO_VALIDATORS_FILESETTINGSPROTOVALIDATOR_H
#define ARRUS_CORE_IO_VALIDATORS_FILESETTINGSPROTOVALIDATOR_H

#include "arrus/common/compiler.h"
#include "arrus/core/common/validation.h"

#include "arrus/core/io/validators/ProbeModelProtoValidator.h"

COMPILER_PUSH_DIAGNOSTIC_STATE
COMPILER_DISABLE_MSVC_WARNINGS(4127)
#include "io/proto/devices/file/FileSettings.pb.h"
COMPILER_POP_DIAGNOSTIC_STATE

namespace arrus::io {

class FileSettingsProtoValidator : public Validator<arrus::proto::FileSettings> {
public:
    explicit FileSettingsProtoValidator(const std::string &name) : Validator(name) {}

    void validate(const arrus::proto::FileSettings &file) override {
        expectTrue("file", !file.filepath().empty(), "Filepath cannot be empty.");
        expectTrue("n_frames", file.n_frames() > 0, "the n_frames value should be > 0");

        if (file.has_probe()) {
            expectTrue("probe_id", !file.has_probe_id(), "Probe Id should not be set (custom probe already set).");
            ProbeModelProtoValidator probeValidator("custom probe");
            auto &probe = file.probe();
            probeValidator.validate(probe);
            copyErrorsFrom(probeValidator);
        } else {
            expectTrue("probe_id", file.has_probe_id(), "Probe id or custom probe def. is required.");
        }
    }
};
}

#endif//ARRUS_CORE_IO_VALIDATORS_FILESETTINGSPROTOVALIDATOR_H
