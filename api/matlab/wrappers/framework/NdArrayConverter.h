#ifndef API_MATLAB_WRAPPERS_FRAMEWORK_NDARRAYCONVERTER_H
#define API_MATLAB_WRAPPERS_FRAMEWORK_NDARRAYCONVERTER_H


#include "api/matlab/wrappers/MexContext.h"
#include "api/matlab/wrappers/convert.h"
#include "arrus/core/api/arrus.h"

#include <boost/bimap.hpp>
#include <mex.hpp>
#include <mexAdapter.hpp>
#include <utility>

namespace arrus::matlab::framework {

using namespace ::arrus::framework;
using namespace ::arrus::matlab::converters;

class NdArrayConverter {
public:
    inline static const std::string MATLAB_FULL_NAME = "arrus.framework.NdArray";

    static NdArrayConverter from(const MexContext::SharedHandle &ctx, const MatlabElementRef &object) {
        const auto placement = ARRUS_MATLAB_GET_CPP_SCALAR(ctx, std::string, placement, object);
        const auto name = ARRUS_MATLAB_GET_CPP_SCALAR(ctx, std::string, name, object);
        const auto value = ctx->createNdArray(getMatlabProperty(ctx, object, "value"), placement, name);
        return NdArrayConverter{
            ctx, value
        };
    }

    static NdArrayConverter from(const MexContext::SharedHandle &ctx, const NdArray &array) {
        return NdArrayConverter{ctx, array};
    }

    NdArrayConverter(const MexContext::SharedHandle &ctx, const NdArray &value)
        : ctx(ctx), value(value) {}

    [[nodiscard]] ::arrus::framework::NdArray toCore() const { return value; }

    [[nodiscard]] ::matlab::data::Array toMatlab() const {
        const auto name = value.getPlacement().toString();
        const auto placement = value.getName();
        return ctx->createObject(
            MATLAB_FULL_NAME,
            {
                ctx->createArray(value),
                ARRUS_MATLAB_GET_MATLAB_STRING_KV_EXPLICIT(ctx, u"placement", placement),
                ARRUS_MATLAB_GET_MATLAB_STRING_KV_EXPLICIT(ctx, u"name", placement),
            }
        );
    }

private:
    MexContext::SharedHandle ctx;
    ::arrus::framework::NdArray value;
};

}// namespace arrus::matlab::framework



#endif//API_MATLAB_WRAPPERS_FRAMEWORK_NDARRAYCONVERTER_H
