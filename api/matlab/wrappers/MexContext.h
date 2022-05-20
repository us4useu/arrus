#ifndef ARRUS_API_MATLAB_WRAPPERS_MEXCONTEXT_H
#define ARRUS_API_MATLAB_WRAPPERS_MEXCONTEXT_H

#include <memory>

#include "arrus/common/compiler.h"
#include "arrus/core/api/common/Logger.h"

COMPILER_PUSH_DIAGNOSTIC_STATE
#pragma warning(disable : 4100 4189 4458 4702)

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
        matlabEngine->feval(
            u"error", 0, std::vector<::matlab::data::Array>({factory.createScalar(msg)}));
    };

    void raiseError(const std::string &errId, const std::string &msg) {
        matlabEngine->feval(
            u"error", 0, std::vector<::matlab::data::Array>({factory.createScalar(errId), factory.createScalar(msg)}));
    };

private:
    ::matlab::data::ArrayFactory factory;
    MatlabEnginePtr matlabEngine;
    Logger::SharedHandle defaultLogger;
};

}// namespace arrus::matlab

#endif//ARRUS_API_MATLAB_WRAPPERS_MEXCONTEXT_H
