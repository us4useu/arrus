%include stdint.i
%include exception.i
%include std_shared_ptr.i
%include std_string.i
%include std_unordered_set.i
%include std_vector.i
%include std_pair.i

// ------------------------------------------ EXCEPTION HANDLING
%exception {
    try {
        $action
    }
    // TODO throw arrus specific exceptions
    catch(const ::arrus::DeviceNotFoundException& e) {
        SWIG_exception(SWIG_ValueError, e.what());
    }
    catch(const ::arrus::IllegalArgumentException& e) {
        SWIG_exception(SWIG_ValueError, e.what());
    }
    catch(const ::arrus::IllegalStateException& e) {
        SWIG_exception(SWIG_RuntimeError, e.what());
    }
    catch(const ::arrus::TimeoutException& e) {
        SWIG_exception(SWIG_RuntimeError, e.what());
    }
    catch(const std::exception &e) {
        SWIG_exception(SWIG_RuntimeError, e.what());
    }
    catch(...) {
        SWIG_exception(SWIG_UnknownError, "Unknown exception.");
    }
}

%module core

%{
#include <memory>
#include <fstream>

#include "arrus/core/api/common/types.h"
#include "arrus/common/logging/impl/Logging.h"
#include "arrus/core/api/io/settings.h"
#include "arrus/core/api/session/Session.h"
#include "arrus/core/api/common/logging.h"
#include "arrus/core/api/devices/us4r/Us4R.h"
#include "arrus/core/api/ops/us4r/TxRxSequence.h"

using namespace ::arrus;
%}

// Naive assumption that only classes starts with capital letter.
%rename("%(undercase)s", notregexmatch$name="^[A-Z].*$") "";
//%rename("%(undercase)s", %$isfunction) "";
//%rename("%(undercase)s", %$isvariable) "";

%nodefaultctor;

// ------------------------------------------ LOGGING
%shared_ptr(arrus::Logger)

%include "arrus/core/api/common/LogSeverity.h"
%include "arrus/core/api/common/Logger.h"

%inline %{
    std::shared_ptr<::arrus::Logging> LOGGING_FACTORY;

    // TODO consider moving the below function to %init
    void initLoggingMechanism(const ::arrus::LogSeverity level) {
        LOGGING_FACTORY = std::make_shared<::arrus::Logging>();
        LOGGING_FACTORY->addClog(level);
        ::arrus::setLoggerFactory(LOGGING_FACTORY);
    }

    void addLogFile(const std::string &filepath, const ::arrus::LogSeverity level) {
        std::shared_ptr<std::ostream> logFileStream =
            // append to the end of the file
            std::make_shared<std::ofstream>(filepath.c_str(), std::ios_base::app);
        LOGGING_FACTORY->addTextSink(logFileStream, level);
    }

    void setClogLevel(const ::arrus::LogSeverity level) {
        LOGGING_FACTORY->setClogLevel(level);
    }

    arrus::Logger::SharedHandle getLogger() {
        ::arrus::Logger::SharedHandle logger = LOGGING_FACTORY->getLogger();
        return logger;
    }
%}

// ------------------------------------------ COMMON
%feature("valuewrapper");

%ignore arrus::Tuple::operator[];
// TO let know swig about any DLL export macros.
%include "arrus/core/api/common/macros.h"
%include "arrus/core/api/common/Tuple.h"
%include "arrus/core/api/common/Interval.h"


// ------------------------------------------ SETTINGS
// TODO wrap std optional
// TODO test creating settings
// TODO test reading settiongs
// TODO feature autodoc
%{
#include "arrus/core/api/devices/us4r/RxSettings.h"
#include "arrus/core/api/devices/us4r/Us4OEMSettings.h"
#include "arrus/core/api/devices/us4r/ProbeAdapterSettings.h"
#include "arrus/core/api/devices/us4r/ProbeAdapterModelId.h"
#include "arrus/core/api/devices/probe/ProbeSettings.h"
#include "arrus/core/api/devices/probe/ProbeModel.h"
#include "arrus/core/api/devices/probe/ProbeModelId.h"
#include "arrus/core/api/devices/us4r/HVSettings.h"
#include "arrus/core/api/devices/us4r/HVModelId.h"
#include "arrus/core/api/devices/us4r/Us4RSettings.h"
#include "arrus/core/api/session/SessionSettings.h"

using namespace ::arrus::devices;
%};

%include "arrus/core/api/devices/us4r/RxSettings.h"
%include "arrus/core/api/devices/us4r/Us4OEMSettings.h"
%ignore operator<<(std::ostream &os, const ProbeAdapterModelId &id);
%include "arrus/core/api/devices/us4r/ProbeAdapterModelId.h"
%include "arrus/core/api/devices/us4r/ProbeAdapterSettings.h"
%ignore operator<<(std::ostream &os, const ProbeModelId &id);
%include "arrus/core/api/devices/probe/ProbeModelId.h"
%include "arrus/core/api/devices/probe/ProbeModel.h"
%include "arrus/core/api/devices/probe/ProbeSettings.h"
%ignore operator<<(std::ostream &os, const HVModelId &id);
%include "arrus/core/api/devices/us4r/HVModelId.h"
%include "arrus/core/api/devices/us4r/HVSettings.h"
%include "arrus/core/api/devices/us4r/Us4RSettings.h"
%include "arrus/core/api/session/SessionSettings.h"

// ------------------------------------------ IO
%include "arrus/core/api/io/settings.h"


// ------------------------------------------ SESSION

%{
#include "arrus/core/api/session/Session.h"
using namespace ::arrus::session;
%};

// TODO consider using unique_ptr anyway
//%feature("novaluewrapper");
//%include "std_unique_ptr.i"
//wrap_unique_ptr(SessionHandle, arrus::session::Session);
%ignore createSession;
%include "arrus/core/api/session/Session.h"
%inline %{
    std::shared_ptr<Session> createSessionSharedHandle(const std::string& filepath) {
        std::shared_ptr<Session> res = createSession(filepath);
        return res;
    }
%};

// ------------------------------------------ DEVICES
// Us4R
%{
#include "arrus/core/api/devices/us4r/Us4R.h"
#include "arrus/core/api/devices/us4r/FrameChannelMapping.h"
#include "arrus/core/api/devices/us4r/HostBuffer.h"
using namespace arrus::devices;
%};

%include "arrus/core/api/devices/us4r/Us4R.h"
%include "arrus/core/api/devices/us4r/FrameChannelMapping.h"
%include "arrus/core/api/devices/us4r/HostBuffer.h"



