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
    constexpr static const float NO_SRI = 0.0f;

    static TxRxSequenceConverter from(const MexContext::SharedHandle &ctx, const MatlabElementRef &object) {
        return TxRxSequenceConverter{ctx,
                                     ARRUS_MATLAB_GET_CPP_OBJECT_VECTOR(ctx, TxRx, TxRxConverter, ops, object),
                                     ARRUS_MATLAB_GET_CPP_VECTOR(ctx, float, tgcCurve, object),
                                     ARRUS_MATLAB_GET_CPP_SCALAR(ctx, float, sri, object),
                                     ARRUS_MATLAB_GET_CPP_SCALAR(ctx, int16, nRepeats, object)};
    }

    static TxRxSequenceConverter from(const MexContext::SharedHandle &ctx, const TxRxSequence &object) {
        std::cout << "Reading from core object" << std::endl;
        return TxRxSequenceConverter{ctx, object.getOps(), object.getTgcCurve(), object.getSri().value_or(NO_SRI), object.getNRepeats()};
    }

    TxRxSequenceConverter(MexContext::SharedHandle ctx, std::vector<TxRx> ops, TGCCurve tgcCurve, float sri, int16 nRepeats)
        : ctx(std::move(ctx)), ops(std::move(ops)), tgcCurve(std::move(tgcCurve)), sri(sri), nRepeats(nRepeats) {}

    [[nodiscard]] ::arrus::ops::us4r::TxRxSequence toCore() const {
        float actualSri = TxRxSequence::NO_SRI;
        if(sri != NO_SRI) {
            // In matlab SRI == 0 is the default value, to not use the SRI.
            actualSri = sri;
        }
        return ::arrus::ops::us4r::TxRxSequence{ops, tgcCurve, actualSri, nRepeats};
    }

    [[nodiscard]] ::matlab::data::Array toMatlab() const {
        std::cout << "Converting to matlab" << std::endl;
        return ctx->createObject(MATLAB_FULL_NAME,
                                 {
                                     ARRUS_MATLAB_GET_MATLAB_OBJECT_VECTOR_KV(ctx, TxRx, TxRxConverter, ops),
                                     ARRUS_MATLAB_GET_MATLAB_VECTOR_KV(ctx, float, tgcCurve),
                                     ARRUS_MATLAB_GET_MATLAB_SCALAR_KV(ctx, float, sri),
                                     ARRUS_MATLAB_GET_MATLAB_SCALAR_KV(ctx, int16, nRepeats)
                                 });
    }

private:
    MexContext::SharedHandle ctx;
    std::vector<TxRx> ops;
    TGCCurve tgcCurve;
    float sri;
    int16 nRepeats;
};

}

#endif//ARRUS_API_MATLAB_WRAPPERS_OPS_US4R_TXRXSEQUENCECONVERTER_H
