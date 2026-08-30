#ifndef API_MATLAB_WRAPPERS_FRAMEWORK_NDSTORAGECONVERTER_H
#define API_MATLAB_WRAPPERS_FRAMEWORK_NDSTORAGECONVERTER_H


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

class NdStorageConverter {
public:
    inline static const std::string MATLAB_FULL_NAME = "arrus.framework.NdStorage";

    static NdStorageConverter from(const MexContext::SharedHandle &ctx, const MatlabElementRef &object) {
        const auto placement = ARRUS_MATLAB_GET_CPP_SCALAR(ctx, std::string, placement, object);
        const auto name = ARRUS_MATLAB_GET_CPP_SCALAR(ctx, std::string, name, object);
        const auto value = ctx->createNdStorage(getMatlabProperty(ctx, object, "value"), placement, name);
        return NdStorageConverter{
            ctx, value
        };
    }

    static NdStorageConverter from(const MexContext::SharedHandle &ctx, const NdStorage &array) {
        return NdStorageConverter{ctx, array};
    }

    NdStorageConverter(const MexContext::SharedHandle &ctx, const NdStorage &value)
        : ctx(ctx), value(value) {}

    [[nodiscard]] ::arrus::framework::NdStorage toCore() const { return value; }

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
    ::arrus::framework::NdStorage value;
};

}// namespace arrus::matlab::framework



#endif//API_MATLAB_WRAPPERS_FRAMEWORK_NDSTORAGECONVERTER_H
