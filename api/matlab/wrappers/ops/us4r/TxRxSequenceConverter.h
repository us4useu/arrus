#ifndef ARRUS_API_MATLAB_WRAPPERS_OPS_US4R_TXRXSEQUENCECONVERTER_H
#define ARRUS_API_MATLAB_WRAPPERS_OPS_US4R_TXRXSEQUENCECONVERTER_H

#include "api/matlab/wrappers/MexContext.h"
#include "api/matlab/wrappers/convert.h"
#include "api/matlab/wrappers/ops/us4r/TxRxConverter.h"
#include "arrus/core/api/arrus.h"

#include <mex.hpp>
#include <mexAdapter.hpp>
#include <utility>

namespace arrus::matlab::ops::us4r {

using namespace ::arrus::ops::us4r;
using namespace ::arrus::matlab::converters;

class TxRxSequenceConverter {
public:
    inline static const std::string MATLAB_FULL_NAME = "arrus.ops.us4r.TxRxSequence";

    static TxRxSequenceConverter from(const MexContext::SharedHandle &ctx, const ::matlab::data::Array &object) {
        return TxRxSequenceConverter{ctx,
                                     ARRUS_MATLAB_GET_CPP_OBJECT_VECTOR(ctx, TxRx, TxRxConverter, txrxs, object),
                                     ARRUS_MATLAB_GET_CPP_VECTOR(ctx, float, tgcCurve, object),
                                     ARRUS_MATLAB_GET_CPP_SCALAR(ctx, float, sri, object)};
    }

    static TxRxSequenceConverter from(const MexContext::SharedHandle &ctx, const TxRxSequence &object) {
        return TxRxSequenceConverter{ctx, object.getOps(), object.getTgcCurve(), object.getSri()};
    }

    TxRxSequenceConverter(MexContext::SharedHandle ctx, std::vector<TxRx> ops, TGCCurve tgcCurve,
                          std::optional<float> sri)
        : ctx(std::move(ctx)), ops(std::move(ops)), tgcCurve(std::move(tgcCurve)), sri(std::move(sri)) {}

    [[nodiscard]] ::arrus::ops::us4r::TxRxSequence toCore() const {
        float actualSri = TxRxSequence::NO_SRI;
        if(sri != 0) {
            // In matlab SRI == 0 is the default value, to not use the SRI.
            actualSri = sri;
        }
        return ::arrus::ops::us4r::TxRxSequence{ops, tgcCurve, actualSri};
    }

    [[nodiscard]] ::matlab::data::Array toMatlab() const {
        return ctx->createObject(MATLAB_FULL_NAME,
                                 {
                                     ARRUS_MATLAB_GET_MATLAB_OBJECT_VECTOR_KV(ctx, TxRx, TxRxConverter, ops),
                                     ARRUS_MATLAB_GET_MATLAB_VECTOR_KV(ctx, float, tgcCurve),
                                     ARRUS_MATLAB_GET_MATLAB_SCALAR_KV(ctx, float, sri)
                                 });
    }

private:
    MexContext::SharedHandle ctx;
    std::vector<TxRx> ops;
    TGCCurve tgcCurve;
    float sri;
};

}

#endif//ARRUS_API_MATLAB_WRAPPERS_OPS_US4R_TXRXSEQUENCECONVERTER_H
