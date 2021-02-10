%include stdint.i
%include exception.i
%include std_shared_ptr.i
%include std_string.i
%include std_unordered_set.i
%include std_vector.i
%include std_pair.i
%include std_optional.i

%{
#include "arrus/core/api/ops/us4r/Rx.h"
#include "arrus/core/api/ops/us4r/Tx.h"
#include "arrus/core/api/ops/us4r/TxRxSequence.h"
#include "arrus/core/api/common/types.h"
using namespace ::arrus;
%};

// TODO try not declaring explicitly the below types
namespace std {
%template(VectorBool) vector<bool>;
%template(VectorFloat) vector<float>;
%template(PairUint32) pair<unsigned, unsigned>;
%template(PairChannelIdx) pair<unsigned short, unsigned short>;

};

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

%module(directors="1") core

%{
#include <memory>
#include <fstream>
#include <iostream>

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
// TODO try enabling underscore option
// However, it is interferring with other swig features, like %template
//%rename("%(undercase)s", notregexmatch$name="^[A-Z].*$") "";

%nodefaultctor;

// TO let know swig about any DLL export macros.
%include "arrus/core/api/common/macros.h"
%include "arrus/core/api/common/types.h"

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
// Turn on globally value wrappers
%feature("valuewrapper", "1");

%ignore arrus::Tuple::operator[];

%include "arrus/core/api/common/Tuple.h"
%include "arrus/core/api/common/Interval.h"

%feature("valuewrapper", "0");

%inline %{
    size_t castToInt(short* ptr) {
        return (size_t)ptr;
    }
%};
// ------------------------------------------ FRAMEWORK

%{
#include "arrus/core/api/framework/NdArray.h"
#include "arrus/core/api/framework/DataBufferSpec.h"
#include "arrus/core/api/framework/Buffer.h"
#include "arrus/core/api/framework/DataBuffer.h"
#include "arrus/core/api/devices/us4r/FrameChannelMapping.h"
using namespace arrus::framework;
using namespace arrus::devices;
%};

%shared_ptr(arrus::devices::FrameChannelMapping);
%shared_ptr(arrus::framework::Buffer);
%shared_ptr(arrus::framework::BufferElement);
%shared_ptr(arrus::framework::DataBuffer);

namespace std {
    %template(FrameChannelMappingElement) pair<unsigned short, arrus::int8>;
};
namespace arrus {
    %template(TupleUint32) Tuple<unsigned int>;
};

%include "arrus/core/api/framework/NdArray.h"
%include "arrus/core/api/devices/us4r/FrameChannelMapping.h"
%include "arrus/core/api/framework/DataBufferSpec.h"
%include "arrus/core/api/framework/Buffer.h"
%include "arrus/core/api/framework/DataBuffer.h"

%feature("director") OnNewDataCallbackWrapper;

%inline %{
class OnNewDataCallbackWrapper {
public:
    OnNewDataCallbackWrapper() {}
    virtual void run(const std::shared_ptr<arrus::framework::BufferElement> element) const {}
    virtual ~OnNewDataCallbackWrapper() {}
};

void registerOnNewDataCallbackFifoLockFreeBuffer(const std::shared_ptr<arrus::framework::Buffer> &buffer, OnNewDataCallbackWrapper& callback) {
    auto fifolockfreeBuffer = std::static_pointer_cast<DataBuffer>(buffer);
    ::arrus::framework::OnNewDataCallback actualCallback = [&](const std::shared_ptr<BufferElement> &ptr) {
            // TODO avoid potential priority inversion here
            PyGILState_STATE gstate = PyGILState_Ensure();
            try {
                callback.run(ptr);
            } catch(const std::exception &e) {
                std::cerr << "Exception: " << e.what() << std::endl;
            } catch(...) {
                std::cerr << "Unhandled exception" << std::endl;
            }
            PyGILState_Release(gstate);
    };
    fifolockfreeBuffer->registerOnNewDataCallback(actualCallback);
}
%};

// ------------------------------------------ SESSION
%{
#include "arrus/core/api/session/UploadConstMetadata.h"
#include "arrus/core/api/session/UploadResult.h"
#include "arrus/core/api/session/Session.h"
using namespace ::arrus::session;

%};
// TODO consider using unique_ptr anyway (https://stackoverflow.com/questions/27693812/how-to-handle-unique-ptrs-with-swig)

%shared_ptr(arrus::session::UploadConstMetadata);
%shared_ptr(arrus::session::Session);
%ignore createSession;


%include "arrus/core/api/session/UploadConstMetadata.h"
%include "arrus/core/api/session/UploadResult.h"
%include "arrus/core/api/session/Session.h"

%inline %{

std::shared_ptr<arrus::session::Session> createSessionSharedHandle(const std::string& filepath) {
    std::shared_ptr<Session> res = createSession(filepath);
    return res;
}

std::shared_ptr<arrus::devices::FrameChannelMapping> getFrameChannelMapping(arrus::session::UploadResult* uploadResult) {
    return uploadResult->getConstMetadata()->get<arrus::devices::FrameChannelMapping>("frameChannelMapping");
}

std::shared_ptr<arrus::framework::DataBuffer> getFifoLockFreeBuffer(arrus::session::UploadResult* uploadResult) {
    auto buffer = std::static_pointer_cast<DataBuffer>(uploadResult->getBuffer());
    return buffer;
}
%};
// ------------------------------------------ DEVICES
// Us4R
%{
#include "arrus/core/api/devices/DeviceId.h"
#include "arrus/core/api/devices/Device.h"
#include "arrus/core/api/devices/DeviceWithComponents.h"
#include "arrus/core/api/devices/us4r/Us4R.h"
#include "arrus/core/api/devices/probe/ProbeModelId.h"
#include "arrus/core/api/devices/probe/Probe.h"
#include "arrus/core/api/devices/probe/ProbeModel.h"

using namespace arrus::devices;
%};

%ignore operator<<(std::ostream &os, const DeviceId &id);
%include "arrus/core/api/devices/DeviceId.h"
%include "arrus/core/api/devices/Device.h"
%include "arrus/core/api/devices/DeviceWithComponents.h"
%include "arrus/core/api/devices/us4r/Us4R.h"
%include "arrus/core/api/devices/probe/ProbeModelId.h"
%include "arrus/core/api/devices/probe/ProbeModel.h"
%include "arrus/core/api/devices/probe/Probe.h"




%inline %{
arrus::devices::Us4R *castToUs4r(arrus::devices::Device *device) {
    auto ptr = dynamic_cast<Us4R*>(device);
    if(!ptr) {
        throw std::runtime_error("Given device is not an us4r handle.");
    }
    return ptr;
}
// TODO(pjarosik) remote the bellow functions when possible

unsigned short getNumberOfElements(const arrus::devices::ProbeModel &probe) {
    const auto &nElements = probe.getNumberOfElements();
    if(nElements.size() > 1) {
        throw ::arrus::IllegalArgumentException("The python API currently cannot be use with 3D probes.");
    }
    return nElements[0];
}

double getPitch(const arrus::devices::ProbeModel &probe) {
    const auto &pitch = probe.getPitch();
    if(pitch.size() > 1) {
        throw ::arrus::IllegalArgumentException("The python API currently cannot be use with 3D probes.");
    }
    return pitch[0];
}
%};



// ------------------------------------------ OPERATIONS


// Us4R
%feature("valuewrapper");
%{
#include "arrus/core/api/ops/us4r/tgc.h"
#include "arrus/core/api/ops/us4r/Pulse.h"
#include "arrus/core/api/ops/us4r/Rx.h"
#include "arrus/core/api/ops/us4r/Tx.h"
#include "arrus/core/api/ops/us4r/TxRxSequence.h"
#include "arrus/core/api/ops/us4r/Scheme.h"
#include <vector>
using namespace arrus::ops::us4r;
%};


%feature("valuewrapper") TxRx;
%include "arrus/core/api/ops/us4r/tgc.h"
%include "arrus/core/api/ops/us4r/Pulse.h"
%include "arrus/core/api/ops/us4r/Rx.h"
%include "arrus/core/api/ops/us4r/Tx.h"
%include "arrus/core/api/ops/us4r/TxRxSequence.h"
%include "arrus/core/api/ops/us4r/Scheme.h"


%include "std_vector.i"
%include "typemaps.i"

namespace std {
%template(TxRxVector) vector<arrus::ops::us4r::TxRx>;
};

%inline %{

void TxRxVectorPushBack(std::vector<arrus::ops::us4r::TxRx> &txrxs,
                        arrus::ops::us4r::TxRx &txrx) {
    txrxs.push_back(txrx);
}

%};


// ------------------------------------------ SETTINGS
// TODO wrap std optional
// TODO test creating settings
// TODO test reading settings
// TODO feature autodoc
// Turn on globally value wrappers
%feature("valuewrapper");
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
