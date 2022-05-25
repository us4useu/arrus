#ifndef ARRUS_API_MATLAB_WRAPPERS_DEVICES_US4R_CONVERTPROBEADAPTERSETTINGS_H
#define ARRUS_API_MATLAB_WRAPPERS_DEVICES_US4R_CONVERTPROBEADAPTERSETTINGS_H

#include "api/matlab/wrappers/MexContext.h"
#include "api/matlab/wrappers/devices/us4r/convertProbeAdapterModelId.h"
#include "arrus/common/format.h"
#include "arrus/core/api/devices/us4r/ProbeAdapterSettings.h"
#include "mex.hpp"

namespace arrus::matlab {
::arrus::devices::ProbeAdapterSettings convertToProbeAdapterSettings(const MexContext::SharedHandle &ctx,
                                                                     const ::matlab::data::Array &object) {

    using namespace arrus::devices;
    auto modelIdArr = getProperty(ctx, object, "modelId");
    ProbeAdapterModelId modelId = convertToProbeAdapterModelId(ctx, modelIdArr);

    auto nChannels = getIntScalar<ChannelIdx>(ctx, object, "nChannels");

    // Channel mapping
    ::matlab::data::TypedArray<double> chMap = getProperty(ctx, object, "channelMapping");

    // convert (2xn) array to list of pairs
    ARRUS_REQUIRES_EQUAL(chMap.getDimensions()[0], 2,
                         IllegalArgumentException("Probe adapter channel mapping "
                                                  "should have exactly two dimensions."));
    ProbeAdapterSettings::ChannelMapping channelMapping(chMap.getDimensions()[1]);

    for (int i = 0; i < chMap.getDimensions()[1]; ++i) {
        double ordinal = chMap[0][i];
        double channel = chMap[1][i];

        ARRUS_MATLAB_REQUIRES_INTEGER_EXCEPTION(
            ordinal,
            IllegalArgumentException(arrus::format("Module's ordinal number should be an integer "
                                                   "(found {} in adapter channel mapping).",
                                                   ordinal)));
        ARRUS_MATLAB_REQUIRES_DATA_TYPE_VALUE_EXCEPTION(
            ordinal, Ordinal,
            IllegalArgumentException(arrus::format("Module's ordinal number should be uint16 "
                                                   "(found {} in adapter channel mapping).",
                                                   ordinal)));

        ARRUS_MATLAB_REQUIRES_INTEGER_EXCEPTION(
            channel,
            IllegalArgumentException(arrus::format("Channel number should be an integer "
                                                   "(found {} in adapter channel mapping).",
                                                   channel)));
        ARRUS_MATLAB_REQUIRES_DATA_TYPE_VALUE_EXCEPTION(
            channel, ChannelIdx,
            IllegalArgumentException(arrus::format("Channel number should be uint16 "
                                                   "(found {} in adapter channel mapping).",
                                                   channel)));
        channelMapping[i] = {(Ordinal) ordinal, (ChannelIdx) channel};
    }
    return ProbeAdapterSettings(modelId, nChannels, channelMapping);
}
}// namespace arrus::matlab

#endif//ARRUS_API_MATLAB_WRAPPERS_DEVICES_US4R_CONVERTPROBEADAPTERSETTINGS_H
