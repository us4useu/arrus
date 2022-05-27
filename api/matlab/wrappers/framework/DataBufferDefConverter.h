#ifndef ARRUS_API_MATLAB_WRAPPERS_FRAMEWORK_DATABUFFERDEFCONVERTER_H
#define ARRUS_API_MATLAB_WRAPPERS_FRAMEWORK_DATABUFFERDEFCONVERTER_H

#include "api/matlab/wrappers/MexContext.h"
#include "api/matlab/wrappers/convert.h"
#include "arrus/core/api/arrus.h"

#include <mex.hpp>
#include <mexAdapter.hpp>
#include <utility>
#include <boost/bimap.hpp>

namespace arrus::matlab::framework {

using namespace ::arrus::framework;
using namespace ::arrus::matlab::converters;

class DataBufferDefConverter {
public:
    inline static const std::string MATLAB_FULL_NAME = "arrus.framework.DataBufferDef";

    static boost::bimap<std::string, DataBufferSpec::Type> &getTypeEnumMap() {
        static boost::bimap<std::string, DataBufferSpec::Type> strToType;
        // NOTE: thread unsafe
        if(strToType.empty()) {
            strToType.left["FIFO"] = DataBufferSpec::Type::FIFO;
        }
        return strToType;
    }

    static DataBufferSpec::Type getType(const std::string &typeStr) {
        try {
            return getTypeEnumMap().left.at(typeStr);
        } catch(const std::out_of_range &e) {
            throw ::arrus::IllegalArgumentException("Unknown enum: " + typeStr);
        }
    }
    static std::string getTypeStr(const DataBufferSpec::Type type) {
        try {
            return getTypeEnumMap().right.at(type);
        } catch(const std::out_of_range &e) {
            throw ::arrus::IllegalArgumentException("Unsupported enum with value: " + std::to_string((int)type));
        }
    }

    static DataBufferDefConverter from(const MexContext::SharedHandle &ctx, const MatlabElementRef &object) {
        return DataBufferDefConverter{
            ctx,
            getType(ARRUS_MATLAB_GET_CPP_SCALAR(ctx, std::string, type, object)),
            ARRUS_MATLAB_GET_CPP_SCALAR(ctx, uint32_t, nElements, object)
        };
    }

    static DataBufferDefConverter from(const MexContext::SharedHandle &ctx, const DataBufferSpec &object) {
        return DataBufferDefConverter{ctx, object.getType(), object.getNumberOfElements()};
    }

    DataBufferDefConverter(MexContext::SharedHandle ctx, DataBufferSpec::Type type, unsigned int nElements)
        : ctx(std::move(ctx)), type(type), nElements(nElements) {}

    [[nodiscard]] ::arrus::framework::DataBufferSpec toCore() const {
        return DataBufferSpec{type, nElements};
    }

    [[nodiscard]] ::matlab::data::Array toMatlab() const {
        return ctx->createObject(MATLAB_FULL_NAME,
                                 {
                                     ARRUS_MATLAB_GET_MATLAB_STRING_KV(ctx, DataBufferDefConverter::getTypeStr(type)),
                                     ARRUS_MATLAB_GET_MATLAB_SCALAR_KV(ctx, uint32_t, nElements)
                                 });
    }
private:
    MexContext::SharedHandle ctx;
    DataBufferSpec::Type type;
    uint32_t nElements;
};

}

#endif//ARRUS_API_MATLAB_WRAPPERS_FRAMEWORK_DATABUFFERDEFCONVERTER_H
