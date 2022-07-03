#ifndef ARRUS_API_MATLAB_WRAPPERS_OPS_US4R_TXRXCONVERTER_H
#define ARRUS_API_MATLAB_WRAPPERS_OPS_US4R_TXRXCONVERTER_H

#include "api/matlab/wrappers/MexContext.h"
#include "api/matlab/wrappers/convert.h"
#include "api/matlab/wrappers/ops/us4r/RxConverter.h"
#include "api/matlab/wrappers/ops/us4r/TxConverter.h"
#include "arrus/core/api/arrus.h"

#include <mex.hpp>
#include <mexAdapter.hpp>
#include <utility>

namespace arrus::matlab::ops::us4r {

using namespace ::arrus::ops::us4r;
using namespace ::arrus::matlab::converters;

class TxRxConverter {
public:
    inline static const std::string MATLAB_FULL_NAME = "arrus.ops.us4r.TxRx";

    static TxRxConverter from(const MexContext::SharedHandle &ctx, const MatlabElementRef &object) {
        return TxRxConverter{ctx,
                             ARRUS_MATLAB_GET_CPP_OBJECT(ctx, Tx, TxConverter, tx, object),
                             ARRUS_MATLAB_GET_CPP_OBJECT(ctx, Rx, RxConverter, rx, object),
                             ARRUS_MATLAB_GET_CPP_SCALAR(ctx, float, pri, object)
        };
    }

    static TxRxConverter from(const MexContext::SharedHandle &ctx, const TxRx &object) {
        return TxRxConverter{ctx, object.getTx(), object.getRx(), object.getPri()};
    }

    TxRxConverter(MexContext::SharedHandle ctx, Tx tx, Rx rx, float pri)
        : ctx(std::move(ctx)), tx(std::move(tx)), rx(std::move(rx)), pri(pri) {}

    [[nodiscard]] ::arrus::ops::us4r::TxRx toCore() const { return ::arrus::ops::us4r::TxRx{tx, rx, pri}; }

    [[nodiscard]] ::matlab::data::Array toMatlab() const {
        return ctx->createObject(MATLAB_FULL_NAME,
                                 {ARRUS_MATLAB_GET_MATLAB_OBJECT_KV(ctx, Tx, TxConverter, tx),
                                  ARRUS_MATLAB_GET_MATLAB_OBJECT_KV(ctx, Rx, RxConverter, rx),
                                  ARRUS_MATLAB_GET_MATLAB_SCALAR_KV(ctx, float, pri)});
    }

private:
    MexContext::SharedHandle ctx;
    Tx tx;
    Rx rx;
    float pri;
};

}

#endif//ARRUS_API_MATLAB_WRAPPERS_OPS_US4R_TXRXCONVERTER_H
