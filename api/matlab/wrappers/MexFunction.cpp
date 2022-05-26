#include <stdexcept>

#include "MexFunction.h"

#include <boost/stacktrace.hpp>
#include <fstream>

#include "arrus/core/api/common/logging.h"
#include "api/matlab/wrappers/ops/us4r/TxRxSequenceConverter.h"

#undef ERROR

MexFunction::MexFunction() {
    mexLock();
//    managers.emplace("Session", new SessionWrapperManager(ctx, "Session"));
}

MexFunction::~MexFunction() { mexUnlock(); }

void MexFunction::operator()(ArgumentList outputs, ArgumentList inputs) {
    try {
        ARRUS_REQUIRES_AT_LEAST(inputs.size(), 2, "The class and method name are missing.");

        MexObjectClassId classId = inputs[0][0];
        MexObjectMethodId methodId = inputs[1][0];

        if (classId == "__global" && methodId == "setLogLevel") {
            // The first call to MexFunction should set console log verbosity, or the default one will be used.
            arrus::LogSeverity sev = getLoggerSeverity(inputs);
            setConsoleLogIfNecessary(sev);
            return;
        }
//        setConsoleLogIfNecessary(arrus::LogSeverity::INFO);
        // Other global functions.
        if (classId == "__global") {
            if (methodId == "addLogFile") {
                ARRUS_REQUIRES_AT_LEAST(inputs.size(), 4, "A path to the log file and logging level are required.");
                std::string filepath = inputs[2][0];
                arrus::LogSeverity level = convertToLogSeverity(inputs[3]);
                std::shared_ptr<std::ostream> logFileStream =
                    std::make_shared<std::ofstream>(filepath.c_str(), std::ios_base::app);
                this->logging->addTextSink(logFileStream, level);
            } else if(methodId == "createExampleObject") {
                auto seq = ::arrus::matlab::ops::us4r::TxRxSequenceConverter::from(ctx, ::arrus::matlab::converters::MatlabElementRef{inputs[2]}).toCore();
                std::cout << "number of ops: " << seq.getOps().size() << std::endl;
                for(auto op : seq.getOps()) {
                    std::cout << "TX: " << std::endl;
                    std::cout << "Aperture: " << std::endl;
                    for(bool v: op.getTx().getAperture()) {
                        std::string vstr = v ? "true" : "false";
                        std::cout << vstr << std::endl;
                    }
                    std::cout << "Delays: " << std::endl;
                    for(float v: op.getTx().getDelays()) {
                        std::cout << v << std::endl;
                    }
                    std::cout << "Pulse: " << std::endl;
                    std::cout << op.getTx().getExcitation().getNPeriods() << std::endl;
                    std::cout << "RX: " << std::endl;
                    std::cout << "Aperture: " << std::endl;
                    for(bool v: op.getRx().getAperture()) {
                        std::string vstr = v ? "true" : "false";
                        std::cout << vstr << std::endl;
                    }
                }
                std::cout << "Properly read the input parameters!" << std::endl;
                std::cout << "Now saving all that to MATLAB objects." << std::endl;
                outputs[0] = ::arrus::matlab::ops::us4r::TxRxSequenceConverter::from(ctx, seq).toMatlab();
                std::cout << "Properly saved to MATLAB!" << std::endl;
            }
            else {
                throw arrus::IllegalArgumentException(arrus::format("Unrecognized global function: {}", methodId));
            }
            return;
        }

        // Class methods.
        // Find appropriate class manager.
        ManagerPtr &manager = managers.at(classId);

        if (methodId == "create") {
            // Constructor.
            // Expected input arguments: classId, 'create', constructor parameters.
            ArgumentList args(inputs.begin() + 2, inputs.end(), inputs.size() - 2);
            auto handle = manager->create(ctx, args);
            outputs[0] = ctx->getArrayFactory().createScalar<MexObjectHandle>(handle);
        } else {

            ARRUS_REQUIRES_AT_LEAST(inputs.size(), 3, "Object handle is missing.");
            MexObjectHandle handle = inputs[2][0];
            if (methodId == "remove") {
                // Class destructor.
                // Expected input arguments: classId, 'remove', object handle.
                manager->remove(handle);
            } else {
                // Any other method.
                // Expected input arguments: classId, methodId, object handle, method parameters.
                ArgumentList args(inputs.begin() + 3, inputs.end(), inputs.size() - 3);

                auto &object = manager->getObject(handle);
                outputs[0] = object->call(methodId, args);
            }
        }
    } catch (const ::arrus::IllegalArgumentException &e) {
        ctx->raiseError("ARRUS:IllegalArgument", e.what());
    } catch (const ::arrus::IllegalStateException &e) {
        ctx->raiseError("ARRUS:IllegalState", e.what());
    } catch (const ::arrus::TimeoutException &e) {
        ctx->raiseError("ARRUS:Timeout", e.what());
    } catch (const std::exception &e) { ctx->raiseError(e.what()); }
}

void MexFunction::setConsoleLogIfNecessary(const arrus::LogSeverity severity) {
    if (logging == nullptr) {
        try {
            this->logging = std::make_shared<arrus::Logging>();
            this->logging->addTextSink(this->matlabOstream, severity, true);
            arrus::Logger::SharedHandle defaultLogger = this->logging->getLogger();
            this->ctx->setDefaultLogger(defaultLogger);
            arrus::setLoggerFactory(logging);
        } catch (const std::exception &e) {
            this->logging = nullptr;
            throw e;
        }
    }
}

arrus::LogSeverity MexFunction::getLoggerSeverity(ArgumentList inputs) {
    ARRUS_REQUIRES_AT_LEAST(inputs.size(), 3, "Log severity level is required.");
    return convertToLogSeverity(inputs[2]);
}

arrus::LogSeverity MexFunction::convertToLogSeverity(const ::matlab::data::Array &severityStr) {
    std::string severity = severityStr[0];
    if (severity == "FATAL") {
        return arrus::LogSeverity::FATAL;
    } else if (severity == "ERROR") {
        return arrus::LogSeverity::ERROR;
    } else if (severity == "WARNING") {
        return arrus::LogSeverity::WARNING;
    } else if (severity == "INFO") {
        return arrus::LogSeverity::INFO;
    } else if (severity == "DEBUG") {
        return arrus::LogSeverity::DEBUG;
    } else if (severity == "TRACE") {
        return arrus::LogSeverity::TRACE;
    } else {
        throw arrus::IllegalArgumentException(arrus::format("Unknown severity level: {}", severity));
    }
}
