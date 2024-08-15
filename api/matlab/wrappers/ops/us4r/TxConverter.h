#ifndef ARRUS_API_MATLAB_WRAPPERS_OPS_US4R_TXCONVERTER_H
#define ARRUS_API_MATLAB_WRAPPERS_OPS_US4R_TXCONVERTER_H

#include "api/matlab/wrappers/MexContext.h"
#include "api/matlab/wrappers/convert.h"
#include "api/matlab/wrappers/ops/us4r/PulseConverter.h"
#include "api/matlab/wrappers/ops/us4r/WaveformConverter.h"
#include "arrus/core/api/arrus.h"

#include <mex.hpp>
#include <mexAdapter.hpp>
#include <utility>

namespace arrus::matlab::ops::us4r {

using namespace ::arrus::ops::us4r;
using namespace ::arrus::matlab::converters;

class TxConverter {
public:
    inline static const std::string MATLAB_FULL_NAME = "arrus.ops.us4r.Tx";

    static TxConverter from(const MexContext::SharedHandle &ctx, const MatlabElementRef &object) {
        ::matlab::data::ObjectArray pulse = getMatlabProperty(ctx, object, "pulse");
        std::string pulseType = convertToString(ctx->getClass(pulse));
        std::optional<Waveform> wf;
        if(pulseType == "arrus.ops.us4r.Pulse") {
            wf = ARRUS_MATLAB_GET_CPP_OBJECT(ctx, Pulse, PulseConverter, pulse, object).toWaveform();
        }
        else if (pulseType == "arrus.ops.us4r.Waveform") {
            wf = ARRUS_MATLAB_GET_CPP_OBJECT(ctx, Waveform, WaveformConverter, pulse, object);
        }
        else {
            throw IllegalArgumentException(format("Unsupported pulse type: {}", pulseType));
        }
        return TxConverter{
            ctx,
            ARRUS_MATLAB_GET_CPP_VECTOR(ctx, bool, aperture, object),
            ARRUS_MATLAB_GET_CPP_VECTOR(ctx, float, delays, object),
            wf.value()
        };
    }

    static TxConverter from(const MexContext::SharedHandle &ctx, const Tx &object) {
        return TxConverter{ctx, object.getAperture(), object.getDelays(), object.getExcitation()};
    }

    TxConverter(const MexContext::SharedHandle &ctx, const std::vector<bool> &aperture,
                const std::vector<float> &delays, const Waveform &waveform)
        : ctx(ctx), aperture(aperture), delays(delays), waveform(waveform) {}

    [[nodiscard]] ::arrus::ops::us4r::Tx toCore() const { return ::arrus::ops::us4r::Tx{aperture, delays, waveform}; }

    [[nodiscard]] ::matlab::data::Array toMatlab() const {
        return ctx->createObject(
            MATLAB_FULL_NAME,
            {ARRUS_MATLAB_GET_MATLAB_VECTOR_KV(ctx, bool, aperture),
             ARRUS_MATLAB_GET_MATLAB_VECTOR_KV(ctx, float, delays),
             ARRUS_MATLAB_GET_MATLAB_OBJECT_KV(ctx, Waveform, WaveformConverter, waveform)});
    }

private:
    MexContext::SharedHandle ctx;
    std::vector<bool> aperture;
    std::vector<float> delays;
    Waveform waveform;
};

}// namespace arrus::matlab::ops::us4r

#endif//ARRUS_API_MATLAB_WRAPPERS_OPS_US4R_TXCONVERTER_H
