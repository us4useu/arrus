#ifndef ARRUS_API_MATLAB_WRAPPERS_DEVICES_PROBE_CONVERTPROBEMODEL_H
#define ARRUS_API_MATLAB_WRAPPERS_DEVICES_PROBE_CONVERTPROBEMODEL_H

#include "arrus/common/asserts.h"
#include "arrus/api/matlab/wrappers/MexContext.h"
#include "arrus/api/matlab/wrappers/devices/probe/convertProbeModelId.h"
#include "arrus/core/api/devices/probe/ProbeModel.h"
#include "mex.hpp"

namespace arrus::matlab {
::arrus::devices::ProbeModel
convertToProbeModel(const MexContext::SharedHandle &ctx,
                    const ::matlab::data::Array &object) {
    using namespace arrus::devices;
    auto modelIdArr = getProperty(ctx, object, "modelId");
    ProbeModelId id = convertToProbeModelId(ctx, modelIdArr);

    using ElementIdxType = ProbeModel::ElementIdxType;

    std::vector<ElementIdxType> nElements = getIntVector<ElementIdxType>(
        ctx, object, "nElements");
    std::vector<double> pitch = getVector<double>(
        ctx, object, "pitch");
    std::vector<float> frequencyRangeVec = getVector<float>(
        ctx, object, "txFrequencyRange");
    std::vector<double> voltageRangeVec = getVector<uint8>(
        ctx, object, "voltageRange");
    ARRUS_REQUIRES_EQUAL(frequencyRangeVec.size(), 2,
                         ::arrus::IllegalArgumentException(
                             "Tx frequency range should contain "
                             "exactly two elements."));

    return ProbeModel(
        id,
        Tuple<ElementIdxType>(nElements),
        Tuple<double>(pitch),
        Interval<float>(frequencyRangeVec[0], frequencyRangeVec[1]),
        Interval<uint8>(voltageRangeVec[0], voltageRangeVec[1])
    );
}
}

#endif //ARRUS_API_MATLAB_WRAPPERS_DEVICES_PROBE_CONVERTPROBEMODEL_H
