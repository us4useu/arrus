#include "MexFunction.h"
#include "arrus/core/api/common/logging.h"

#include <boost/stacktrace.hpp>

#undef ERROR

MexFunction::MexFunction() {
    mexLock();
    managers.emplace("Session",
                     new SessionWrapperManager(mexContext, "Session"));
}

MexFunction::~MexFunction() {
    mexUnlock();
}

void MexFunction::operator()(ArgumentList outputs, ArgumentList inputs) {
    try {
        ARRUS_REQUIRES_AT_LEAST(inputs.size(), 2,
                                "The class and method name are missing.");

        MexObjectClassId classId = inputs[0][0];
        MexObjectMethodId methodId = inputs[1][0];

        if(classId == "__global" && methodId == "setConsoleLogger") {
            // The first call to MexFunction should set console log
            // verbosity, or the default one will be used.
            arrus::LogSeverity sev = getLoggerSeverity(inputs);
            setConsoleLogIfNecessary(sev);
            return;
        }
        setConsoleLogIfNecessary(arrus::LogSeverity::INFO);
        // Other global functions.
        if(classId == "__global") {
            // TODO
        }

        ManagerPtr &manager = managers.at(classId);

        if(methodId == "create") {
            ArgumentList args(inputs.begin() + 2, inputs.end(),
                              inputs.size() - 2);
            auto handle = manager->create(mexContext, args);
            outputs[0] = mexContext->getArrayFactory().createScalar<MexObjectHandle>(
                handle);
        } else {
            ARRUS_REQUIRES_AT_LEAST(inputs.size(), 3,
                                    "Object handle is missing.");
            MexObjectHandle handle = inputs[2][0];

            if(methodId == "remove") {
                manager->remove(handle);
            } else {
                ArgumentList args(inputs.begin() + 3, inputs.end(),
                                  inputs.size() - 3);

                auto &object = manager->getObject(handle);
                outputs[0] = object->call(methodId, args);
            }
        }
    }
    catch(const std::exception &e) {
        mexContext->raiseError(e.what());
    }

}

void MexFunction::setConsoleLogIfNecessary(const arrus::LogSeverity severity) {
    if(logging == nullptr) {
        try {
            this->logging = std::make_shared<arrus::Logging>();
            this->logging->addTextSink(this->matlabOstream, severity, true);
            arrus::Logger::SharedHandle defaultLogger = this->logging->getLogger();
            this->mexContext->setDefaultLogger(defaultLogger);
            arrus::setLoggerFactory(logging);
        } catch(const std::exception &e) {
            this->logging = nullptr;
            throw e;
        }

    }
}

arrus::LogSeverity MexFunction::getLoggerSeverity(ArgumentList inputs) {
    ARRUS_REQUIRES_AT_LEAST(inputs.size(), 3,
                            "Log severity level is required.");
    std::string severity = inputs[2][0];
    if(severity == "FATAL") {
        return arrus::LogSeverity::FATAL;
    } else if(severity == "ERROR") {
        return arrus::LogSeverity::ERROR;
    } else if(severity == "WARNING") {
        return arrus::LogSeverity::WARNING;
    } else if(severity == "INFO") {
        return arrus::LogSeverity::INFO;
    } else if(severity == "DEBUG") {
        return arrus::LogSeverity::DEBUG;
    } else if(severity == "TRACE") {
        return arrus::LogSeverity::TRACE;
    } else {
        throw arrus::IllegalArgumentException(
            arrus::format("Unknown severity level: {}", severity));
    }
}

