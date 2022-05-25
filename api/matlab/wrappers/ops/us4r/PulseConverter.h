#ifndef ARRUS_API_MATLAB_WRAPPERS_OPS_US4R_PULSECONVERTER_H
#define ARRUS_API_MATLAB_WRAPPERS_OPS_US4R_PULSECONVERTER_H

#include "api/matlab/wrappers/MexContext.h"
#include "arrus/core/api/arrus.h"
#include "api/matlab/wrappers/convert.h"

#include <mex.hpp>
#include <mexAdapter.hpp>

namespace arrus::matlab::ops::us4r {

using namespace ::arrus::ops::us4r;
using namespace ::arrus::matlab::converters;


class PulseConverter {
public:
    inline static const std::u16string MATLAB_FULL_NAME = u"arrus.ops.us4r.Pulse";

    static PulseConverter from(const MexContext::SharedHandle &ctx, const ::matlab::data::Array &object) {
        return PulseConverter{
            ctx,
            ARRUS_MATLAB_GET_CPP_REQUIRED_SCALAR(ctx, float, centerFrequency, object),
            ARRUS_MATLAB_GET_CPP_REQUIRED_SCALAR(ctx, float, nPeriods, object),
            ARRUS_MATLAB_GET_CPP_REQUIRED_SCALAR(ctx, bool, inverse, object)
        };
    }

    static PulseConverter from(const MexContext::SharedHandle &ctx, const Pulse &object) {
        return PulseConverter{ctx, object.getCenterFrequency(), object.getNPeriods(), object.isInverse() };
    }

    PulseConverter(MexContext::SharedHandle ctx, float centerFrequency, float nPeriods, bool inverse)
        : ctx(std::move(ctx)), centerFrequency(centerFrequency), nPeriods(nPeriods), inverse(inverse) {}

    [[nodiscard]] ::arrus::ops::us4r::Pulse toCore() const {
        return ::arrus::ops::us4r::Pulse{centerFrequency, nPeriods, inverse};
    }

    [[nodiscard]] ::matlab::data::Array toMatlab() const {
        return ctx->createObject(
            MATLAB_FULL_NAME,
            {
                ARRUS_MATLAB_GET_MATLAB_SCALAR(ctx, float, centerFrequency),
                ARRUS_MATLAB_GET_MATLAB_SCALAR(ctx, float, nPeriods),
                ARRUS_MATLAB_GET_MATLAB_SCALAR(ctx, bool, inverse)
            }
        );
    }

private:
    MexContext::SharedHandle ctx;
    float centerFrequency;
    float nPeriods;
    bool inverse;
};

}// namespace arrus::matlab::ops::us4r

#endif//ARRUS_API_MATLAB_WRAPPERS_OPS_US4R_PULSECONVERTER_H
