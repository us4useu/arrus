#ifndef ARRUS_API_MATLAB_WRAPPERS_DEVICES_US4R_CONVERTUS4OEMSETTINGS_H
#define ARRUS_API_MATLAB_WRAPPERS_DEVICES_US4R_CONVERTUS4OEMSETTINGS_H

#include "arrus/api/matlab/wrappers/MexContext.h"
#include "arrus/api/matlab/wrappers/convert.h"
#include "arrus/api/matlab/wrappers/asserts.h"
#include "arrus/api/matlab/wrappers/devices/us4r/convertRxSettings.h"
#include "arrus/core/api/devices/us4r/Us4OEMSettings.h"
#include "mex.hpp"

namespace arrus::matlab {
arrus::devices::Us4OEMSettings
convertToUs4OEMSettings(const MexContext::SharedHandle &ctx,
                        const ::matlab::data::Array &object) {
    using namespace arrus::devices;
    // Channel mapping.
    ::matlab::data::TypedArray<double> channelMappingArr = getProperty(
        ctx, object, "channelMapping");
    ARRUS_REQUIRES_ALL_DATA_TYPE_VALUE(channelMappingArr, ChannelIdx,
                                       "us4oem channel mapping.");
    ARRUS_MATLAB_REQUIRES_ALL_INTEGER(channelMappingArr);
    std::vector<ChannelIdx> channelMapping =
        convertToVector<ChannelIdx>(channelMappingArr);
    // Active channel groups.
    ::matlab::data::TypedArray<double> activeChGrArr = getProperty(
        ctx, object, "activeChannelGroups");
    ARRUS_MATLAB_REQUIRES_ALL_BINARY(activeChGrArr);
    BitMask activeChannelGroups = convertToVector<bool>(activeChGrArr);

    // rx settings.
    ::matlab::data::Array rxSettingsArr = getProperty(ctx, object,
                                                      "rxSettings");
    RxSettings rxSettings = convertToRxSettings(ctx, rxSettingsArr);
    return Us4OEMSettings(channelMapping, activeChannelGroups,
                          rxSettings);
}
}

#endif //ARRUS_API_MATLAB_WRAPPERS_DEVICES_US4R_CONVERTUS4OEMSETTINGS_H
