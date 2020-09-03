#ifndef ARRUS_CORE_IO_VALIDATORS_PROBEADAPTERMODELPROTOVALIDATOR_H
#define ARRUS_CORE_IO_VALIDATORS_PROBEADAPTERMODELPROTOVALIDATOR_H

#include "arrus/common/compiler.h"
#include "arrus/core/common/validation.h"

COMPILER_PUSH_DIAGNOSTIC_STATE
COMPILER_DISABLE_MSVC_WARNINGS(4127)

#include "io/proto/devices/us4r/ProbeAdapterModel.pb.h"

namespace arrus::io {

class ProbeAdapterModelProtoValidator
    : public Validator<arrus::proto::ProbeAdapterModel> {
public:

    explicit ProbeAdapterModelProtoValidator(const std::string &componentName)
        : Validator(componentName) {}

    void validate(const arrus::proto::ProbeAdapterModel &obj) override {
        bool hasChannelMappings = obj.has_channels_mapping();
        bool hasChannelMappingsRegions = !obj.channel_mapping_regions().empty();
        expectTrue("channel mapping",
                   !(hasChannelMappings ^ hasChannelMappingsRegions),
                   "Exactly one of the following should be set for "
                   "probe adapter model: (channel mappings, channel "
                   "mapping regions)");
    }

};

}

#endif //ARRUS_CORE_IO_VALIDATORS_PROBEADAPTERMODELPROTOVALIDATOR_H
