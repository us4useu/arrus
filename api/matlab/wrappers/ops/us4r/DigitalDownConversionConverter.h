#ifndef ARRUS_API_MATLAB_WRAPPERS_OPS_US4R_DIGITALDOWNCONVERSIONCONVERTER_H
#define ARRUS_API_MATLAB_WRAPPERS_OPS_US4R_DIGITALDOWNCONVERSIONCONVERTER_H

#include "api/matlab/wrappers/MexContext.h"
#include "api/matlab/wrappers/convert.h"
#include "arrus/core/api/arrus.h"

#include <utility>

namespace arrus::matlab::ops::us4r {

using namespace ::arrus::ops::us4r;
using namespace ::arrus::matlab::converters;

class DigitalDownConversionConverter {
public:
    inline static const std::string MATLAB_FULL_NAME = "arrus.ops.us4r.DigitalDownConversion";

    static DigitalDownConversionConverter from(const MexContext::SharedHandle &ctx, const MatlabElementRef &object) {
        return DigitalDownConversionConverter {
            ctx,
            ARRUS_MATLAB_GET_CPP_SCALAR(ctx, float, demodulationFrequency, object),
            ARRUS_MATLAB_GET_CPP_SCALAR(ctx, float, decimationFactor, object),
            ARRUS_MATLAB_GET_CPP_VECTOR(ctx, float, firCoefficients, object)
        };
    }

    static DigitalDownConversionConverter from(const MexContext::SharedHandle &ctx, const DigitalDownConversion &object) {
        Span<float> coeffs{object.getFirCoefficients()};
        std::vector<float> coeffsCopy(coeffs.size(), 0.0f);
        for(int i = 0; i < coeffs.size(); ++i) {
            coeffsCopy[i] = coeffs[i];
        }
        return DigitalDownConversionConverter{ctx, object.getDemodulationFrequency(), object.getDecimationFactor(),
            coeffsCopy};
    }

    DigitalDownConversionConverter(const MexContext::SharedHandle &ctx, float demodulationFrequency,
                                   float decimationFactor, std::vector<float> firCoefficients)
        : ctx(ctx), demodulationFrequency(demodulationFrequency), decimationFactor(decimationFactor),
          firCoefficients(std::move(firCoefficients)) {}

    [[nodiscard]] ::arrus::ops::us4r::DigitalDownConversion toCore() const {
        return DigitalDownConversion{demodulationFrequency, firCoefficients, decimationFactor};
    }

    [[nodiscard]] ::matlab::data::Array toMatlab() const {
        return ctx->createObject(
            MATLAB_FULL_NAME,
            {
             ARRUS_MATLAB_GET_MATLAB_SCALAR_KV(ctx, float, demodulationFrequency),
             ARRUS_MATLAB_GET_MATLAB_SCALAR_KV(ctx, float, decimationFactor),
             ARRUS_MATLAB_GET_MATLAB_VECTOR_KV(ctx, float, firCoefficients)
            });
    }

private:
    MexContext::SharedHandle ctx;
    float demodulationFrequency;
    float decimationFactor;
    std::vector<float> firCoefficients;
};
}// namespace arrus::matlab::ops::us4r

#endif//ARRUS_API_MATLAB_WRAPPERS_OPS_US4R_DIGITALDOWNCONVERSIONCONVERTER_H
