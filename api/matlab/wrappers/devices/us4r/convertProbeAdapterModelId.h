#ifndef ARRUS_API_MATLAB_WRAPPERS_DEVICES_US4R_CONVERTPROBEADAPTERMODELID_H
#define ARRUS_API_MATLAB_WRAPPERS_DEVICES_US4R_CONVERTPROBEADAPTERMODELID_H

#include "arrus/api/matlab/wrappers/MexContext.h"
#include "arrus/core/api/devices/us4r/ProbeAdapterModelId.h"
#include "mex.hpp"

namespace arrus::matlab {
    ::arrus::devices::ProbeAdapterModelId
    convertToProbeAdapterModelId(const MexContext::SharedHandle &ctx,
                                  const ::matlab::data::Array &object) {
        using namespace arrus::devices;
        std::string name = getProperty(ctx, object, "name")[0];
        std::string manuf = getProperty(ctx, object, "manufacturer")[0];
        return ProbeAdapterModelId(name, manuf);
    }
}

#endif //ARRUS_API_MATLAB_WRAPPERS_DEVICES_US4R_CONVERTPROBEADAPTERMODELID_H
