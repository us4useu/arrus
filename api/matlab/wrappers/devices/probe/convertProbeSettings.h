#ifndef ARRUS_API_MATLAB_WRAPPERS_DEVICES_PROBE_CONVERTPROBESETTINGS_H
#define ARRUS_API_MATLAB_WRAPPERS_DEVICES_PROBE_CONVERTPROBESETTINGS_H

#include "arrus/api/matlab/wrappers/MexContext.h"
#include "arrus/core/api/devices/probe/ProbeSettings.h"
#include "arrus/api/matlab/wrappers/convert.h"
#include "convertProbeModel.h"
#include "mex.hpp"

namespace arrus::matlab {
ProbeSettings
convertToProbeSettings(const MexContext::SharedHandle &ctx,
                       const ::matlab::data::Array &object) {
    auto modelArr = getProperty(ctx, object, "probeModel");
    ProbeModel model = convertToProbeModel(ctx, modelArr);

    using ElementType = ProbeModel::ElementIdxType;

    std::vector<ElementType> channelMapping = getIntVector<ElementType>(
        ctx, object, "nElements");

    return ProbeSettings(model, channelMapping);
}
}

#endif //ARRUS_API_MATLAB_WRAPPERS_DEVICES_PROBE_CONVERTPROBESETTINGS_H
