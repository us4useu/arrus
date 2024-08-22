#ifndef ARRUS_API_MATLAB_WRAPPERS_OPS_US4R_WAVEFORMCONVERTER_H
#define ARRUS_API_MATLAB_WRAPPERS_OPS_US4R_WAVEFORMCONVERTER_H

#include "api/matlab/wrappers/MexContext.h"
#include "arrus/core/api/arrus.h"
#include "api/matlab/wrappers/convert.h"
#include "api/matlab/wrappers/ops/us4r/WaveformSegmentConverter.h"

#include <mex.hpp>
#include <mexAdapter.hpp>

namespace arrus::matlab::ops::us4r {

using namespace ::arrus::ops::us4r;
using namespace ::arrus::matlab::converters;


class WaveformConverter {
public:
    inline static const std::string MATLAB_FULL_NAME = "arrus.ops.us4r.Waveform";

    static WaveformConverter from(const MexContext::SharedHandle &ctx, const MatlabElementRef &object) {
        return WaveformConverter{
            ctx,
            ARRUS_MATLAB_GET_CPP_OBJECT_VECTOR(ctx, WaveformSegment, WaveformSegmentConverter, segments, object),
            ARRUS_MATLAB_GET_CPP_VECTOR(ctx, size_t, nRepeats, object)
        };
    }

    static WaveformConverter from(const MexContext::SharedHandle &ctx, const Waveform &object) {
        return WaveformConverter{ctx, object.getSegments(), object.getNRepetitions()};
    }

    WaveformConverter(const MexContext::SharedHandle &ctx,
                      const std::vector<::arrus::ops::us4r::WaveformSegment> &segments,
                      const std::vector<size_t> &nRepeats)
        : ctx(ctx), segments(segments), nRepeats(nRepeats) {}

    [[nodiscard]] ::arrus::ops::us4r::Waveform toCore() const {
        return ::arrus::ops::us4r::Waveform{segments, nRepeats};
    }

    [[nodiscard]] ::matlab::data::Array toMatlab() const {
        return ctx->createObject(
            MATLAB_FULL_NAME,
            {
                ARRUS_MATLAB_GET_MATLAB_OBJECT_VECTOR_KV(ctx, WaveformSegment, WaveformSegmentConverter, segments),
                ARRUS_MATLAB_GET_MATLAB_VECTOR_KV(ctx, size_t, nRepeats)
            }
        );
    }

private:
    MexContext::SharedHandle ctx;
    std::vector<::arrus::ops::us4r::WaveformSegment> segments;
    std::vector<size_t> nRepeats;
};

}// namespace arrus::matlab::ops::us4r

#endif//ARRUS_API_MATLAB_WRAPPERS_OPS_US4R_WAVEFORMCONVERTER_H
