#ifndef ARRUS_CORE_IO_VALIDATORS_GPUSETTINGSPROTOVALIDATOR_H
#define ARRUS_CORE_IO_VALIDATORS_GPUSETTINGSPROTOVALIDATOR_H

#include "arrus/common/compiler.h"
#include "arrus/core/common/validation.h"

COMPILER_PUSH_DIAGNOSTIC_STATE
COMPILER_DISABLE_MSVC_WARNINGS(4127)

#include "io/proto/devices/us4r/GpuSettings.pb.h"

COMPILER_POP_DIAGNOSTIC_STATE

namespace arrus::io {

class GpuSettingsProtoValidator : public Validator<arrus::proto::GpuSettings> {
public:
    explicit GpuSettingsProtoValidator(const std::string &name) : Validator(name) {}

    void validate(const arrus::proto::GpuSettings &obj) override {
        // Memory limit percentage is optional, but if provided, it should be a valid percentage
        if (obj.memory_limit_percentage() > 0.0f) {
            validateMemoryLimitPercentage(obj.memory_limit_percentage());
        }
        
        // Validate use_memory_pool field (boolean validation is implicit)
        // The field is optional and defaults to false in protobuf
        // No additional validation needed for boolean fields
    }

private:
    void validateMemoryLimitPercentage(float memoryLimitPercentage) {
        // Check if the memory limit percentage is within valid range (0.0 to 1.0)
        if (memoryLimitPercentage < 0.0f || memoryLimitPercentage > 1.0f) {
            expectTrue("memory_limit_percentage", false, 
                      "Memory limit percentage must be between 0.0 and 1.0. Got: " + std::to_string(memoryLimitPercentage));
            return;
        }
        
        // Check if the percentage is reasonable (not too small)
        if (memoryLimitPercentage < 0.01f) {
            expectTrue("memory_limit_percentage", false, 
                      "Memory limit percentage is too small. Minimum allowed is 0.01 (1%). Got: " + std::to_string(memoryLimitPercentage));
            return;
        }
        
        // Check if the percentage is reasonable (not too large)
        if (memoryLimitPercentage > 0.95f) {
            expectTrue("memory_limit_percentage", false, 
                      "Memory limit percentage is too large. Maximum allowed is 0.95 (95%). Got: " + std::to_string(memoryLimitPercentage));
            return;
        }
    }
};

}

#endif //ARRUS_CORE_IO_VALIDATORS_GPUSETTINGSPROTOVALIDATOR_H 