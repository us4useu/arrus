#ifndef ARRUS_API_MATLAB_WRAPPERS_DEVICES_US4R_CONVERTUS4RSETTINGS_H
#define ARRUS_API_MATLAB_WRAPPERS_DEVICES_US4R_CONVERTUS4RSETTINGS_H

#include <vector>

#include "arrus/api/matlab/wrappers/MexContext.h"
#include "arrus/api/matlab/wrappers/convert.h"
#include "arrus/api/matlab/wrappers/devices/us4r/convertUs4OEMSettings.h"
#include "arrus/api/matlab/wrappers/devices/us4r/convertProbeAdapterSettings.h"
#include "arrus/api/matlab/wrappers/devices/us4r/convertProbeSettings.h"
#include "arrus/api/matlab/wrappers/devices/us4r/convertRxSettings.h"
#include "arrus/core/api/devices/us4r/Us4RSettings.h"
#include "mex.hpp"

namespace arrus::matlab {

Us4RSettings convertToUs4RSettings(const MexContext::SharedHandle &ctx,
                                   const ::matlab::data::Array &object) {
    auto us4OEMProp = getProperty(ctx, object, "us4OEMSettings");
    if(!us4OEMProp.isEmpty()) {
        ::std::vector<Us4OEMSettings> res;
        for(unsigned i = 0; i < us4OEMProp.getNumberOfElements(); ++i) {
            ::matlab::data::Array arr = us4OEMProp[i];
            res.emplace_back(convertToUs4OEMSettings(ctx, arr));
        }
        return Us4RSettings(res);
    } else {
        auto adapterArr = getRequiredScalar(ctx, object,
                                            "probeAdapterSettings");
        ProbeAdapterSettings adapterSettings = convertToProbeAdapterSettings(
            ctx, adapterArr);

        auto probeArr = getRequiredScalar(ctx, object, "probeSettings");
        ProbeSettings probeSettings = convertToProbeSettings(ctx, probeArr);

        auto rxArr = getRequiredScalar(ctx, object, "rxSettings");
        auto rxSettings = convertToRxSettings(ctx, rxArr);

        return Us4RSettings(adapterSettings, probeSettings, rxSettings);
    }
}
}

#endif //ARRUS_API_MATLAB_WRAPPERS_DEVICES_US4R_CONVERTUS4RSETTINGS_H
