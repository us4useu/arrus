#ifndef ARRUS_API_MATLAB_WRAPPERS_DEVICES_PROBE_CONVERTPROBEMODELID_H
#define ARRUS_API_MATLAB_WRAPPERS_DEVICES_PROBE_CONVERTPROBEMODELID_H

#include "arrus/core/api/devices/probe/ProbeModelId.h"
#include "arrus/api/matlab/wrappers/MexContext.h"
#include "arrus/core/api/devices/us4r/ProbeAdapterModelId.h"
#include "arrus/api/matlab/wrappers/convert.h"
#include "mex.hpp"

namespace arrus::matlab {
ProbeModelId
convertToProbeModelId(const MexContext::SharedHandle &ctx,
                      const ::matlab::data::Array &object) {
    std::string name = getProperty(ctx, object, "name")[0];
    std::string manuf = getProperty(ctx, object, "manufacturer")[0];
    return ProbeModelId(name, manuf);
}
}

#endif //ARRUS_API_MATLAB_WRAPPERS_DEVICES_PROBE_CONVERTPROBEMODELID_H
