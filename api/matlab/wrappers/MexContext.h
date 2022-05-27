#ifndef ARRUS_API_MATLAB_WRAPPERS_MEXCONTEXT_H
#define ARRUS_API_MATLAB_WRAPPERS_MEXCONTEXT_H

#include <memory>

#include "arrus/common/compiler.h"
#include "arrus/common/format.h"
#include "arrus/core/api/common/Logger.h"
#include "arrus/core/api/common/exceptions.h"

COMPILER_PUSH_DIAGNOSTIC_STATE
#pragma warning(disable : 4100 4189 4458 4702)// MSVC

#include <MatlabDataArray/ArrayFactory.hpp>
#include <mex.hpp>
#include <utility>

COMPILER_POP_DIAGNOSTIC_STATE

namespace arrus::matlab {

class MexContext {
public:
    using MatlabEnginePtr = std::shared_ptr<::matlab::engine::MATLABEngine>;

    using SharedHandle = std::shared_ptr<MexContext>;

    explicit MexContext(MatlabEnginePtr matlabEngine) : matlabEngine(std::move(matlabEngine)) {}

    [[nodiscard]] ::matlab::data::ArrayFactory &getArrayFactory() { return factory; }

    MatlabEnginePtr &getMatlabEngine() { return matlabEngine; }

    void setDefaultLogger(const Logger::SharedHandle &logger) { this->defaultLogger = logger; }

    void log(LogSeverity severity, const std::string &msg) {
        if (this->defaultLogger != nullptr) {
            this->defaultLogger->log(severity, msg);
        } else {
            matlabEngine->feval(u"disp", 0, std::vector<::matlab::data::Array>({factory.createScalar(msg)}));
        }
    }
    void logTrace(const std::string &msg) { log(LogSeverity::TRACE, msg); }

    void logDebug(const std::string &msg) { log(LogSeverity::DEBUG, msg); }

    void logInfo(const std::string &msg) { log(LogSeverity::INFO, msg); }

    void logWarning(const std::string &msg) { log(LogSeverity::WARNING, msg); }

    void logError(const std::string &msg) { log(LogSeverity::ERROR, msg); }

    void logFatal(const std::string &msg) { log(LogSeverity::FATAL, msg); }

    void raiseError(const std::string &msg) {
        matlabEngine->feval(u"error", 0, std::vector<::matlab::data::Array>({factory.createScalar(msg)}));
    };

    void raiseError(const std::string &errId, const std::string &msg) {
        matlabEngine->feval(
            u"error", 0, std::vector<::matlab::data::Array>({factory.createScalar(errId), factory.createScalar(msg)}));
    };

    ::matlab::data::Array createObject(const std::string &typeId, const std::vector<::matlab::data::Array> &params) {
        try {
            return matlabEngine->feval(typeId, 1, params)[0];
        } catch (const std::exception &e) {
            throw ::arrus::IllegalArgumentException(
                ::arrus::format("Exception while creating object '{}': {}", typeId, e.what()));
        }
    }

    template<typename T>::matlab::data::TypedArray<T> createScalar(const T &value) {
        try {
            return getArrayFactory().createScalar<T>(value);
        } catch (const std::exception &e) {
            throw ::arrus::IllegalArgumentException(
                ::arrus::format("Exception while creating scalar array '{}': {}", value, e.what()));
        }
    }

    ::matlab::data::TypedArray<::matlab::data::MATLABString> createScalarString(const ::matlab::data::MATLABString &v) {
        try {
            return getArrayFactory().createScalar(v);
        } catch (const std::exception &e) {
            throw ::arrus::IllegalArgumentException(
                ::arrus::format("Exception while creating scalar string array: {}", e.what()));
        }
    }

    ::matlab::data::TypedArray<::matlab::data::MATLABString> createScalarString(const ::std::string &v) {
        try {
            return getArrayFactory().createScalar(v);
        } catch (const std::exception &e) {
            throw ::arrus::IllegalArgumentException(
                ::arrus::format("Exception while creating scalar string array: {}", e.what()));
        }
    }

    template<typename T>::matlab::data::TypedArray<T> createVector(const std::vector<T> &value) {
        try {
            ::matlab::data::ArrayDimensions dimensions = {1, value.size()};
            return getArrayFactory().createArray(dimensions, std::begin(value), std::end(value));
        } catch (const std::exception &e) {
            throw ::arrus::IllegalArgumentException(::arrus::format(
                "Exception while creating scalar string array '{}': {}", ::arrus::toString(value), e.what()));
        }
    }

private:
    ::matlab::data::ArrayFactory factory;
    MatlabEnginePtr matlabEngine;
    Logger::SharedHandle defaultLogger;
};

}// namespace arrus::matlab

#endif//ARRUS_API_MATLAB_WRAPPERS_MEXCONTEXT_H
