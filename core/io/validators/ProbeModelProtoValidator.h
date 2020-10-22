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
        auto &freqRange = obj.tx_frequency_range();
        expectTrue("tx_frequency_range", freqRange.begin() <= freqRange.end(),
                   "tx freq range begin should be <= tx freq range end.");

        auto &voltageRange = obj.voltage_range();
        expectTrue("voltage_range", voltageRange.begin() <= voltageRange.end(),
                   "voltage range begin should be <= voltage range end.");

        expectDataType<uint8>("voltage_range.begin", voltageRange.begin());
        expectDataType<uint8>("voltage_range.end", voltageRange.end());
    }
};

}


#endif //ARRUS_CORE_IO_VALIDATORS_PROBEMODELPROTOVALIDATOR_H
