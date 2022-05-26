#ifndef ARRUS_API_MATLAB_WRAPPERS_OPS_US4R_RXCONVERTER_H
#define ARRUS_API_MATLAB_WRAPPERS_OPS_US4R_RXCONVERTER_H

#include "api/matlab/wrappers/MexContext.h"
#include "api/matlab/wrappers/convert.h"
#include "api/matlab/wrappers/ops/us4r/PulseConverter.h"
#include "arrus/core/api/arrus.h"

#include <mex.hpp>
#include <mexAdapter.hpp>
#include <utility>

namespace arrus::matlab::ops::us4r {

using namespace ::arrus::ops::us4r;
using namespace ::arrus::matlab::converters;

class RxConverter {
public:
    inline static const std::string MATLAB_FULL_NAME = "arrus.ops.us4r.Rx";

    static RxConverter from(const MexContext::SharedHandle &ctx, const ::matlab::data::Array &object) {
        return RxConverter{ctx, ARRUS_MATLAB_GET_CPP_VECTOR(ctx, bool, aperture, object),
                           ARRUS_MATLAB_GET_CPP_PAIR(ctx, uint32_t, sampleRange, object),
                           ARRUS_MATLAB_GET_CPP_SCALAR(ctx, uint32_t, downsamplingFactor, object),
                           ARRUS_MATLAB_GET_CPP_PAIR(ctx, uint16_t, sampleRange, object)};
    }

    static RxConverter from(const MexContext::SharedHandle &ctx, const Rx &object) {
        return RxConverter{ctx, object.getAperture(), object.getSampleRange(), object.getDownsamplingFactor(),
                           object.getPadding()};
    }

    RxConverter(MexContext::SharedHandle ctx, std::vector<bool> aperture,
                std::pair<unsigned int, unsigned int> sampleRange, unsigned int downsamplingFactor,
                std::pair<unsigned short, unsigned short> padding)
        : ctx(std::move(ctx)), aperture(std::move(aperture)), sampleRange(std::move(sampleRange)),
          downsamplingFactor(downsamplingFactor), padding(std::move(padding)) {}

    [[nodiscard]] ::arrus::ops::us4r::Rx toCore() const {
        return ::arrus::ops::us4r::Rx{aperture, sampleRange, downsamplingFactor, padding};
    }

    [[nodiscard]] ::matlab::data::Array toMatlab() const {
        return ctx->createObject(MATLAB_FULL_NAME,
                                 {ARRUS_MATLAB_GET_MATLAB_VECTOR_KV(ctx, bool, aperture),
                                  ARRUS_MATLAB_GET_MATLAB_VECTOR_KV(ctx, uint32_t, sampleRange),
                                  ARRUS_MATLAB_GET_MATLAB_SCALAR_KV(ctx, uint32_t, downsamplingFactor),
                                  ARRUS_MATLAB_GET_MATLAB_VECTOR_KV(ctx, float, padding)});
    }

private:
    MexContext::SharedHandle ctx;
    std::vector<bool> aperture;
    std::pair<unsigned, unsigned> sampleRange;
    unsigned downsamplingFactor;
    std::pair<unsigned short, unsigned short> padding;
};

}

#endif//ARRUS_API_MATLAB_WRAPPERS_OPS_US4R_RXCONVERTER_H
