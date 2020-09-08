#ifndef ARRUS_API_MATLAB_WRAPPERS_MEXCONTEXT_H
#define ARRUS_API_MATLAB_WRAPPERS_MEXCONTEXT_H

#include <memory>

#include "arrus/common/compiler.h"
#include "arrus/core/api/common/Logger.h"

COMPILER_PUSH_DIAGNOSTIC_STATE
#pragma warning(disable: 4100 4189 4458 4702)

#include <mex.hpp>
#include <MatlabDataArray/ArrayFactory.hpp>
#include <utility>

COMPILER_POP_DIAGNOSTIC_STATE

namespace arrus::matlab {

    class MexContext {
    public:
        using MatlabEnginePtr = std::shared_ptr<::matlab::engine::MATLABEngine>;

        using SharedHandle = std::shared_ptr<MexContext>;

        explicit MexContext(MatlabEnginePtr matlabEngine)
            : matlabEngine(std::move(matlabEngine)) {}

        [[nodiscard]] ::matlab::data::ArrayFactory &getArrayFactory() {
            return factory;
        }

        MatlabEnginePtr &getMatlabEngine() {
            return matlabEngine;
        }

        void setDefaultLogger(const Logger::SharedHandle &logger) {
            this->defaultLogger = logger;
        }

        void log(LogSeverity severity, const std::string &msg) {
            if(this->defaultLogger != nullptr) {
                this->defaultLogger->log(severity, msg);
            }
            else {
                matlabEngine->feval(u"disp", 0,
                                    std::vector<::matlab::data::Array>(
                                        {factory.createScalar(msg)}));
            }
        }

        void logInfo(const std::string &msg) {
            log(LogSeverity::INFO, msg);
        }

        void raiseError(const std::string &msg) {
            // TODO(pjarosik) add exception name as defined in common dropbox paper
            matlabEngine->feval(u"error", 0,
                                std::vector<::matlab::data::Array>(
                                    {factory.createScalar(msg)}));
        };

    private:
        ::matlab::data::ArrayFactory factory;
        MatlabEnginePtr matlabEngine;
        Logger::SharedHandle defaultLogger;
    };


}

#endif //ARRUS_API_MATLAB_WRAPPERS_MEXCONTEXT_H
