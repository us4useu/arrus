#ifndef ARRUS_API_MATLAB_WRAPPERS_DEVICES_PROBE_CONVERTPROBEMODEL_H
#define ARRUS_API_MATLAB_WRAPPERS_DEVICES_PROBE_CONVERTPROBEMODEL_H

#include "arrus/api/matlab/wrappers/MexContext.h"
#include "arrus/api/matlab/wrappers/devices/probe/convertProbeModelId.h"
#include "arrus/core/api/devices/probe/ProbeModel.h"
#include "mex.hpp"

namespace arrus::matlab {
ProbeModel
convertToProbeModel(const MexContext::SharedHandle &ctx,
                    const ::matlab::data::Array &object) {
    auto modelIdArr = getProperty(ctx, object, "modelId");
    ProbeModelId id = convertToProbeModelId(ctx, modelIdArr);

    using ElementIdxType = ProbeModel::ElementIdxType;

    std::vector<ElementIdxType> nElements = getIntVector<ElementIdxType>(
        ctx, object, "nElements");
    std::vector<double> pitch = getVector<double>(
        ctx, object, "pitch");
    std::vector<double> frequencyRangeVec = getVector<double>(
        ctx, object, "txFrequencyRange");
    ARRUS_REQUIRES_EQUAL(frequencyRangeVec.size(), 2,
                         IllegalArgumentException(
                             "Tx frequency range should contain "
                             "exactly two elements."));

    return ProbeModel(
        id,
        Tuple<ElementIdxType>(nElements),
        Tuple<double>(pitch),
        Interval<double>(frequencyRangeVec[0], frequencyRangeVec[1])
    );
}
}

#endif //ARRUS_API_MATLAB_WRAPPERS_DEVICES_PROBE_CONVERTPROBEMODEL_H
