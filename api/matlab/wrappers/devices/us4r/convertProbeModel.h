#ifndef ARRUS_API_MATLAB_WRAPPERS_DEVICES_US4R_CONVERTPROBEMODEL_H
#define ARRUS_API_MATLAB_WRAPPERS_DEVICES_US4R_CONVERTPROBEMODEL_H

#include "arrus/api/matlab/wrappers/MexContext.h"
#include "arrus/core/api/devices/probe/ProbeModel.h"
#include "mex.hpp"

namespace arrus::matlab {
    ProbeModel
    convertToProbeModel(const MexContext::SharedHandle &ctx,
                        const ::matlab::data::Array &object) {
        // TODO implement
    }
}

#endif //ARRUS_API_MATLAB_WRAPPERS_DEVICES_US4R_CONVERTPROBEMODEL_H
