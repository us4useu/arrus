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
        using namespace arrus::devices;
        bool hasChannelMapping = obj.has_channel_mapping();
        bool hasChannelMappingsRegions = !obj.channel_mapping_regions().empty();

        // Data types
        expectDataType<ChannelIdx>("n_channels", obj.n_channels());
        if(hasChannelMapping) {
            auto const &ordinals = obj.channel_mapping().us4oems();
            auto const &channels = obj.channel_mapping().channels();
            expectAllDataType<Ordinal>(
                "channel_mapping.us4oems", ordinals);
            expectAllDataType<ChannelIdx>(
                "channel_mapping.channels", channels);
        }
        if(hasChannelMappingsRegions) {
            for(auto const &region: obj.channel_mapping_regions()) {
                expectTrue(
                    "channel_mapping_regions",
                    region.has_region() ^ !region.channels().empty(),
                    "Exactly one of the following should be provided: region, channels"
                );

                expectDataType<Ordinal>("channel_mapping_regions.us4oem",
                                        region.us4oem());
                if(region.has_region()) {

                    expectDataType<ChannelIdx>(
                        "channel_mapping_regions.region.begin",
                        region.region().begin());

                    expectDataType<ChannelIdx>(
                        "channel_mapping_regions.region.end",
                        region.region().end());

                } else {
                    expectAllDataType<ChannelIdx>(
                        "channel_mapping_regions.channels", region.channels());
                }

            }
        }

        // Semantic
        expectTrue("channel mapping",
                   hasChannelMapping ^ hasChannelMappingsRegions,
                   "Exactly one of the following should be set for "
                   "probe adapter model: (channel mappings, channel "
                   "mapping regions)");
    }

};

}

#endif //ARRUS_CORE_IO_VALIDATORS_PROBEADAPTERMODELPROTOVALIDATOR_H
