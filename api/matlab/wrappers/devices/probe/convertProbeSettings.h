#ifndef ARRUS_API_MATLAB_WRAPPERS_DEVICES_PROBE_CONVERTPROBESETTINGS_H
#define ARRUS_API_MATLAB_WRAPPERS_DEVICES_PROBE_CONVERTPROBESETTINGS_H

#include "api/matlab/wrappers/MexContext.h"
#include "api/matlab/wrappers/convert.h"
#include "arrus/core/api/devices/probe/ProbeSettings.h"
#include "convertProbeModel.h"
#include "mex.hpp"

namespace arrus::matlab {
::arrus::devices::ProbeSettings convertToProbeSettings(const MexContext::SharedHandle &ctx,
                                                       const ::matlab::data::Array &object) {
    using namespace arrus::devices;
    auto modelArr = getProperty(ctx, object, "probeModel");
    ProbeModel model = convertToProbeModel(ctx, modelArr);

    using ElementType = ProbeModel::ElementIdxType;

    std::vector<ElementType> channelMapping = getIntVector<ElementType>(ctx, object, "channelMapping");

    return ProbeSettings(model, channelMapping);
}
}// namespace arrus::matlab

#endif//ARRUS_API_MATLAB_WRAPPERS_DEVICES_PROBE_CONVERTPROBESETTINGS_H
