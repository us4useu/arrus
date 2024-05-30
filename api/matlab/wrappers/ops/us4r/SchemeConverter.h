#ifndef ARRUS_API_MATLAB_WRAPPERS_OPS_US4R_SCHEMECONVERTER_H
#define ARRUS_API_MATLAB_WRAPPERS_OPS_US4R_SCHEMECONVERTER_H

#include "TxRxSequenceConverter.h"
#include "api/matlab/wrappers/MexContext.h"
#include "api/matlab/wrappers/convert.h"
#include "api/matlab/wrappers/framework/DataBufferDefConverter.h"
#include "api/matlab/wrappers/ops/us4r/DigitalDownConversionConverter.h"
#include "arrus/core/api/arrus.h"

#include <boost/bimap.hpp>
#include <mex.hpp>
#include <mexAdapter.hpp>
#include <utility>

namespace arrus::matlab::ops::us4r {

using namespace ::arrus::ops::us4r;
using namespace ::arrus::matlab::converters;

class SchemeConverter {
public:
    inline static const std::string MATLAB_FULL_NAME = "arrus.ops.us4r.Scheme";
    typedef boost::bimap<std::string, Scheme::WorkMode> enummap;
    typedef enummap::value_type enummapElement;

    static enummap &getWorkModeEnumMap() {
        static enummap strToEnum;
        // NOTE: thread unsafe
        if (strToEnum.empty()) {
            strToEnum.insert(enummapElement{"ASYNC", Scheme::WorkMode::ASYNC});
            strToEnum.insert(enummapElement{"SYNC", Scheme::WorkMode::SYNC});
            strToEnum.insert(enummapElement{"HOST", Scheme::WorkMode::HOST});
            strToEnum.insert(enummapElement{"MANUAL", Scheme::WorkMode::MANUAL});
        }
        return strToEnum;
    }

    static Scheme::WorkMode getWorkMode(const std::string &modeStr) {
        try {
            return getWorkModeEnumMap().left.at(modeStr);
        } catch (const std::out_of_range &e) { throw ::arrus::IllegalArgumentException("Unknown enum: " + modeStr); }
    }
    static std::string getWorkModeStr(const Scheme::WorkMode mode) {
        try {
            return getWorkModeEnumMap().right.at(mode);
        } catch (const std::out_of_range &e) {
            throw ::arrus::IllegalArgumentException("Unsupported enum with value: " + std::to_string((int) mode));
        }
    }

    static SchemeConverter from(const MexContext::SharedHandle &ctx, const MatlabElementRef &object) {
        return SchemeConverter{
            ctx, ARRUS_MATLAB_GET_CPP_OBJECT(ctx, TxRxSequence, TxRxSequenceConverter, txRxSequence, object),
            ARRUS_MATLAB_GET_CPP_SCALAR(ctx, uint16_t, rxBufferSize, object),
            ARRUS_MATLAB_GET_CPP_OBJECT(ctx, framework::DataBufferSpec,
                                        ::arrus::matlab::framework::DataBufferDefConverter, outputBuffer, object),
            getWorkMode(ARRUS_MATLAB_GET_CPP_SCALAR(ctx, std::string, workMode, object)),
            ARRUS_MATLAB_GET_CPP_OPTIONAL_OBJECT(ctx, DigitalDownConversion, DigitalDownConversionConverter,
                                                 digitalDownConversion, object),
        };
    }

    static SchemeConverter from(const MexContext::SharedHandle &ctx, const Scheme &object) {
        return SchemeConverter{ctx, object.getTxRxSequence(), object.getRxBufferSize(), object.getOutputBuffer(),
                               object.getWorkMode(), object.getDigitalDownConversion()};
    }

    SchemeConverter(MexContext::SharedHandle ctx, TxRxSequence txRxSequence, uint16 rxBufferSize,
                    const framework::DataBufferSpec &outputBuffer, Scheme::WorkMode workMode,
                    std::optional<DigitalDownConversion> digitalDownConversion)
        : ctx(std::move(ctx)), txRxSequence(std::move(txRxSequence)), rxBufferSize(rxBufferSize),
          outputBuffer(outputBuffer), workMode(workMode), digitalDownConversion(std::move(digitalDownConversion)) {}

    [[nodiscard]] ::arrus::ops::us4r::Scheme toCore() const {
        if(digitalDownConversion.has_value()) {
            return Scheme{txRxSequence, rxBufferSize, outputBuffer, workMode, digitalDownConversion.value()};
        }
        else {
            return Scheme{txRxSequence, rxBufferSize, outputBuffer, workMode};
        }
    }

    [[nodiscard]] ::matlab::data::Array toMatlab() const {
        return ctx->createObject(
            MATLAB_FULL_NAME,
            {ARRUS_MATLAB_GET_MATLAB_OBJECT_KV(ctx, TxRxSequence, TxRxSequenceConverter, txRxSequence),
             ARRUS_MATLAB_GET_MATLAB_SCALAR_KV(ctx, uint16_t, rxBufferSize),
             ARRUS_MATLAB_GET_MATLAB_OBJECT_KV(ctx, framework::DataBufferSpec, ::arrus::matlab::framework::DataBufferDefConverter, outputBuffer),
             ARRUS_MATLAB_GET_MATLAB_STRING_KV_EXPLICIT(ctx, u"workMode", SchemeConverter::getWorkModeStr(workMode)),
             ARRUS_MATLAB_GET_MATLAB_OBJECT_KV(ctx, DigitalDownConversion, DigitalDownConversionConverter, digitalDownConversion),
            });
    }

private:
    MexContext::SharedHandle ctx;
    TxRxSequence txRxSequence;
    uint16 rxBufferSize;
    ::arrus::framework::DataBufferSpec outputBuffer;
    Scheme::WorkMode workMode;
    std::optional<DigitalDownConversion> digitalDownConversion;
};
}// namespace arrus::matlab::ops::us4r

#endif//ARRUS_API_MATLAB_WRAPPERS_OPS_US4R_SCHEMECONVERTER_H
