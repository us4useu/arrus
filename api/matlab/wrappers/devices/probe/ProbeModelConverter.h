#ifndef ARRUS_API_MATLAB_WRAPPERS_DEVICES_PROBE_PROBE_MODEL_H
#define ARRUS_API_MATLAB_WRAPPERS_DEVICES_PROBE_PROBE_MODEL_H

#include "api/matlab/wrappers/MexContext.h"
#include "arrus/core/api/arrus.h"
#include "api/matlab/wrappers/convert.h"
#include "api/matlab/wrappers/devices/probe/ProbeModelIdConverter.h"

#include <mex.hpp>
#include <mexAdapter.hpp>
#include <utility>

namespace arrus::matlab::devices::probe {

using namespace ::arrus::devices;
using namespace ::arrus::matlab::converters;


class ProbeModelConverter {
public:
    inline static const std::string MATLAB_FULL_NAME = "arrus.devices.probe.ProbeModel";

    // MATLAB object -> ProbeModelConverter.
    static ProbeModelConverter from(const MexContext::SharedHandle &ctx, const MatlabElementRef &object) {
        return ProbeModelConverter{
            ctx,
            ARRUS_MATLAB_GET_CPP_OBJECT(ctx, ProbeModelId, ProbeModelIdConverter, modelId, object),
            ARRUS_MATLAB_GET_CPP_VECTOR(ctx, ProbeModel::ElementIdxType, nElements, object),
            ARRUS_MATLAB_GET_CPP_VECTOR(ctx, double, pitch, object),
            ARRUS_MATLAB_GET_CPP_PAIR(ctx, float, txFrequencyRange, object),
            ARRUS_MATLAB_GET_CPP_PAIR(ctx, Voltage, voltageRange, object),
            ARRUS_MATLAB_GET_CPP_SCALAR(ctx, double, curvatureRadius, object),
            ARRUS_MATLAB_GET_CPP_SCALAR(ctx, double, lensDelay, object)
        };
    }

    // C++ API object -> ProbeModelConverter.
    static ProbeModelConverter from(const MexContext::SharedHandle &ctx, const ProbeModel &object) {
        return ProbeModelConverter{
            ctx,
            object.getModelId(),
            object.getNumberOfElements().getValues(),
            object.getPitch().getValues(),
            object.getTxFrequencyRange().asPair(),
            object.getVoltageRange().asPair(),
            object.getCurvatureRadius(),
            object.getLensDelay()
        };
    }

    ProbeModelConverter(
        MexContext::SharedHandle ctx,
        ProbeModelId modelId,
        const std::vector<ProbeModel::ElementIdxType> &numberOfElements,
        const std::vector<double> &pitch,
        const std::pair<float, float> &txFrequencyRange,
        const std::pair<Voltage, Voltage> &voltageRange,
        double curvatureRadius,
        double lensDelay
    )
        : ctx(std::move(ctx)), modelId(std::move(modelId)), nElements(numberOfElements), pitch(pitch),
          txFrequencyRange(txFrequencyRange), voltageRange(voltageRange), curvatureRadius(curvatureRadius), 
          lensDelay(lensDelay) {}

    // ProbeModelConverter -> C++ API object.
    [[nodiscard]] ::arrus::devices::ProbeModel toCore() const {
        return ::arrus::devices::ProbeModel{
            modelId, nElements,
            pitch,
            txFrequencyRange,
            voltageRange,
            curvatureRadius,
            lensDelay
        };
    }

    // ProbeModelConverter -> MATLAB API object.
    [[nodiscard]] ::matlab::data::Array toMatlab() const {
        return ctx->createObject(
            MATLAB_FULL_NAME,
            {
                ARRUS_MATLAB_GET_MATLAB_OBJECT_KV(ctx, ProbeModelId, ProbeModelIdConverter, modelId),
                ARRUS_MATLAB_GET_MATLAB_VECTOR_KV_EXPLICIT(ctx, ProbeModel::ElementIdxType, nElements, nElements.getValues()),
                ARRUS_MATLAB_GET_MATLAB_VECTOR_KV_EXPLICIT(ctx, double, pitch, pitch.getValues()),
                ARRUS_MATLAB_GET_MATLAB_VECTOR_KV_EXPLICIT(ctx, float, txFrequencyRange, txFrequencyRange.asPair()),
                ARRUS_MATLAB_GET_MATLAB_VECTOR_KV_EXPLICIT(ctx, Voltage, voltageRange, voltageRange.asPair()),
                ARRUS_MATLAB_GET_MATLAB_SCALAR_KV(ctx, double, curvatureRadius),
                ARRUS_MATLAB_GET_MATLAB_SCALAR_KV(ctx, double, lensDelay),
            }
        );
    }

private:
    MexContext::SharedHandle ctx;
    ProbeModelId modelId;
    Tuple<ProbeModel::ElementIdxType> nElements;
    Tuple<double> pitch;
    Interval<float> txFrequencyRange;
    Interval<Voltage> voltageRange;
    double curvatureRadius;
    double lensDelay;
};

} //

#endif//ARRUS_API_MATLAB_WRAPPERS_DEVICES_PROBE_PROBE_MODEL_H
