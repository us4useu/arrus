#ifndef ARRUS_CORE_IO_VALIDATORS_PROBETOADAPTERCONNECTIONPROTOVALIDATOR_H
#define ARRUS_CORE_IO_VALIDATORS_PROBETOADAPTERCONNECTIONPROTOVALIDATOR_H

#include "arrus/common/compiler.h"
#include "arrus/core/common/validation.h"

COMPILER_PUSH_DIAGNOSTIC_STATE
COMPILER_DISABLE_MSVC_WARNINGS(4127)

#include "io/proto/devices/us4r/ProbeToAdapterConnection.pb.h"

COMPILER_POP_DIAGNOSTIC_STATE

namespace arrus::io {

class ProbeToAdapterConnectionProtoValidator
    : public Validator<arrus::proto::ProbeToAdapterConnection> {
    public:

    using Validator::Validator;

    void validate(const proto::ProbeToAdapterConnection &obj) override {
        bool hasChannelMapping = !obj.channel_mapping().empty();
        bool hasMappingIntervals = !obj.channel_mapping_ranges().empty();

        // data types
        if(hasChannelMapping) {
            expectAllDataType<ChannelIdx>(
                "channel_mapping",
                obj.channel_mapping());
        }

        if(hasMappingIntervals) {
            for(auto const &range : obj.channel_mapping_ranges()) {
                expectTrue("range", range.begin() <= range.end(),
                            "should be begin <= end.");
                expectDataType<ChannelIdx>("region.begin", range.begin());
                expectDataType<ChannelIdx>("region.end", range.end());
            }
        }

        // semantic
        expectTrue("probe_to_adapter_connection",
                   hasChannelMapping ^ hasMappingIntervals,
                   "Exactly one of the following should set: "
                   "channel_mapping, channel_mappign_ranges"
        );
    }

};


}

#endif //ARRUS_CORE_IO_VALIDATORS_PROBETOADAPTERCONNECTIONPROTOVALIDATOR_H
