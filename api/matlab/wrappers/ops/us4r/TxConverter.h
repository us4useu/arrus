#ifndef ARRUS_API_MATLAB_WRAPPERS_OPS_US4R_TXCONVERTER_H
#define ARRUS_API_MATLAB_WRAPPERS_OPS_US4R_TXCONVERTER_H

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

class TxConverter {
public:
    inline static const std::string MATLAB_FULL_NAME = "arrus.ops.us4r.Tx";

    static TxConverter from(const MexContext::SharedHandle &ctx, const MatlabElementRef &object) {
        return TxConverter{
            ctx,
            ARRUS_MATLAB_GET_CPP_VECTOR(ctx, bool, aperture, object),
            ARRUS_MATLAB_GET_CPP_VECTOR(ctx, float, delays, object),
            ARRUS_MATLAB_GET_CPP_OBJECT(ctx, Pulse, PulseConverter, pulse, object)
        };
    }

    static TxConverter from(const MexContext::SharedHandle &ctx, const Tx &object) {
        return TxConverter{ctx, object.getAperture(), object.getDelays(), object.getExcitation()};
    }

    TxConverter(MexContext::SharedHandle ctx, std::vector<bool> aperture, std::vector<float> delays,
                const Pulse &pulse)
        : ctx(std::move(ctx)), aperture(std::move(aperture)), delays(std::move(delays)), pulse(pulse) {}

    [[nodiscard]] ::arrus::ops::us4r::Tx toCore() const { return ::arrus::ops::us4r::Tx{aperture, delays, pulse}; }

    [[nodiscard]] ::matlab::data::Array toMatlab() const {
        return ctx->createObject(
            MATLAB_FULL_NAME,
            {ARRUS_MATLAB_GET_MATLAB_VECTOR_KV(ctx, bool, aperture),
             ARRUS_MATLAB_GET_MATLAB_VECTOR_KV(ctx, float, delays),
             ARRUS_MATLAB_GET_MATLAB_OBJECT_KV(ctx, Pulse, PulseConverter, pulse)});
    }

private:
    MexContext::SharedHandle ctx;
    std::vector<bool> aperture;
    std::vector<float> delays;
    Pulse pulse;
};

}// namespace arrus::matlab::ops::us4r

#endif//ARRUS_API_MATLAB_WRAPPERS_OPS_US4R_TXCONVERTER_H
