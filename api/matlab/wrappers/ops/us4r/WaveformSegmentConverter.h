#ifndef ARRUS_API_MATLAB_WRAPPERS_OPS_US4R_WAVEFORMSEGMENTCONVERTER_H
#define ARRUS_API_MATLAB_WRAPPERS_OPS_US4R_WAVEFORMSEGMENTCONVERTER_H

#include "arrus/core/api/arrus.h"
#include "api/matlab/wrappers/MexContext.h"
#include "api/matlab/wrappers/convert.h"

#include <mex.hpp>
#include <mexAdapter.hpp>


namespace arrus::matlab::ops::us4r {

using namespace ::arrus::ops::us4r;
using namespace ::arrus::matlab::converters;


class WaveformSegmentConverter {

public:
    inline static const std::string MATLAB_FULL_NAME = "arrus.ops.us4r.WaveformSegment";

    static WaveformSegmentConverter from(const MexContext::SharedHandle &ctx, const MatlabElementRef &object) {
        return WaveformSegmentConverter{
            ctx,
            ARRUS_MATLAB_GET_CPP_VECTOR(ctx, float, duration, object),
            ARRUS_MATLAB_GET_CPP_VECTOR(ctx, int8, state, object)
        };
    }

    static WaveformSegmentConverter from(const MexContext::SharedHandle &ctx, const WaveformSegment &object) {
        return WaveformSegmentConverter{ctx, object.getDuration(), object.getState() };
    }

    WaveformSegmentConverter(const MexContext::SharedHandle &ctx, const std::vector<float> &duration,
                             const std::vector<int8> &state)
        : ctx(ctx), duration(duration), state(state) {}

    [[nodiscard]] ::arrus::ops::us4r::WaveformSegment toCore() const {
        return ::arrus::ops::us4r::WaveformSegment{duration, state};
    }

    [[nodiscard]] ::matlab::data::Array toMatlab() const {
        return ctx->createObject(
            MATLAB_FULL_NAME,
            {
                ARRUS_MATLAB_GET_MATLAB_VECTOR_KV(ctx, float, duration),
                ARRUS_MATLAB_GET_MATLAB_VECTOR_KV(ctx, int8, state)
            }
        );
    }

private:
    MexContext::SharedHandle ctx;
    std::vector<float> duration;
    std::vector<int8> state;
};


}


#endif//ARRUS_API_MATLAB_WRAPPERS_OPS_US4R_WAVEFORMSEGMENTCONVERTER_H
