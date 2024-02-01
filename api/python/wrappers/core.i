%include stdint.i
%include exception.i
%include std_shared_ptr.i
%include std_string.i
%include std_unordered_set.i
%include std_vector.i
%include std_pair.i

%inline %{
/**
 * A class that keeps "unlocked" GIL state in the RAII style.
 * That is, you will release the GIL when this object is created,
 * and obtain the GIL again when the object is deleted.
*/
class ArrusPythonGILUnlock {
public:
    ArrusPythonGILUnlock()
        :state(PyEval_SaveThread()) {}

    ~ArrusPythonGILUnlock() {
        PyEval_RestoreThread(state);
    }
private:
    PyThreadState* state;
};
%}

%{
#include <string>
#include <optional>
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
%template(VectorUInt16) vector<unsigned short>;
%template(VectorUInt8) vector<unsigned char>;
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


// std typemaps
// std::optional
%typemap(in) std::optional<arrus::uint16> %{
    if($input == Py_None) {
        $1 = std::optional<arrus::uint16>();
    }
    else {
        long value = PyLong_AsLong($input);
        // TODO(refactor) extract safe cast macro
        if(value > std::numeric_limits<arrus::uint16>::max() || value < std::numeric_limits<arrus::uint16>::min()) {
            std::string errorMsg = "Value '" + std::to_string(value) + "' should be in range: ["
                + std::to_string(std::numeric_limits<arrus::uint16>::min())
                + ", " + std::to_string(std::numeric_limits<arrus::uint16>::max()) + "]";
            PyErr_SetString(PyExc_ValueError, errorMsg.c_str());
            return NULL;
        }
        $1 = std::optional<arrus::uint16>(value);
    }
%}
%typemap(out) boost::optional<arrus::uint16> %{
    if($1) {
        $result = PyLong_FromLong(*$1);
    }
    else {
        $result = Py_None;
        Py_INCREF(Py_None);
    }
%}


%module(directors="1") core

%{
#include <memory>
#include <fstream>
#include <iostream>

#include "arrus/core/api/common/types.h"
#include "arrus/core/api/io/settings.h"
#include "arrus/core/api/session/Session.h"
#include "arrus/core/api/common/logging.h"
#include "arrus/core/api/devices/us4r/Us4OEM.h"
#include "arrus/core/api/devices/us4r/Us4R.h"
#include "arrus/core/api/ops/us4r/TxRxSequence.h"
#include "arrus/core/api/devices/File.h"

using namespace ::arrus;
%}

// Naive assumption that only classes starts with capital letter.
// TODO try enabling underscore option
// However, it is interferring with other swig features, like %template
//%rename("%(undercase)s", notregexmatch$name="^[A-Z].*$") "";

%nodefaultctor;

// TO let know swig about any DLL export macros.
#define __attribute__(x)

%include "arrus/core/api/common/macros.h"
%include "arrus/core/api/common/types.h"

// ------------------------------------------ LOGGING
%shared_ptr(arrus::Logger)

%include "arrus/core/api/common/LogSeverity.h"
%include "arrus/core/api/common/Logger.h"

%inline %{
    ::arrus::Logging* LOGGING_FACTORY;

    // TODO consider moving the below function to %init
    void initLoggingMechanism(const ::arrus::LogSeverity level) {
        LOGGING_FACTORY = ::arrus::useDefaultLoggerFactory();
        LOGGING_FACTORY->addClog(level);
    }

    void addLogFile(const std::string &filepath, const ::arrus::LogSeverity level) {
        std::shared_ptr<std::ostream> logFileStream =
            // append to the end of the file
            std::make_shared<std::ofstream>(filepath.c_str(), std::ios_base::app);
        LOGGING_FACTORY->addOutputStream(logFileStream, level);
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
%include "arrus/core/api/common/Span.h"
%include "arrus/core/api/ops/us4r/DigitalDownConversion.h"
%include "arrus/core/api/common/Parameters.h"

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

namespace arrus {
    %template(TupleUint32) Tuple<unsigned int>;
    %template(TupleSizeT) Tuple<size_t>;
    %template(IntervalFloat) Interval<float>;
};

%ignore arrus::framework::NdArray::NdArray;
%include "arrus/core/api/framework/NdArray.h"

%include "arrus/core/api/devices/us4r/FrameChannelMapping.h"
%include "arrus/core/api/framework/DataBufferSpec.h"
%include "arrus/core/api/framework/Buffer.h"
%include "arrus/core/api/framework/DataBuffer.h"

%feature("director") OnNewDataCallbackWrapper;
%feature("director") OnBufferOverflowCallbackWrapper;

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

class OnBufferOverflowCallbackWrapper {
public:
    OnBufferOverflowCallbackWrapper() {}
    virtual void run() const {}
    virtual ~OnBufferOverflowCallbackWrapper() {}
};

void registerOnBufferOverflowCallback(const std::shared_ptr<arrus::framework::Buffer> &buffer, OnBufferOverflowCallbackWrapper& callback) {
    auto fifolockfreeBuffer = std::static_pointer_cast<DataBuffer>(buffer);
    ::arrus::framework::OnOverflowCallback actualCallback = [&]() {
        PyGILState_STATE gstate = PyGILState_Ensure();
        try {
            callback.run();
        } catch(const std::exception &e) {
            std::cerr << "Exception: " << e.what() << std::endl;
        } catch(...) {
            std::cerr << "Unhandled exception" << std::endl;
        }
        PyGILState_Release(gstate);
    };
    fifolockfreeBuffer->registerOnOverflowCallback(actualCallback);
}
%};

// ------------------------------------------ SESSION
%{
#include "arrus/core/api/session/Metadata.h"
#include "arrus/core/api/session/UploadResult.h"
#include "arrus/core/api/session/Session.h"
using namespace ::arrus::session;

%};
// TODO consider using unique_ptr anyway (https://stackoverflow.com/questions/27693812/how-to-handle-unique-ptrs-with-swig)

%shared_ptr(arrus::session::Metadata);
%shared_ptr(arrus::session::Session);
%ignore createSession;


%include "arrus/core/api/session/Metadata.h"
%include "arrus/core/api/session/UploadResult.h"
%include "arrus/core/api/session/Session.h"

%inline %{

std::shared_ptr<arrus::session::Session> createSessionSharedHandle(const std::string& filepath) {
    std::shared_ptr<Session> res = createSession(filepath);
    return res;
}

std::shared_ptr<arrus::devices::FrameChannelMapping> getFrameChannelMapping(size_t arrayId, arrus::session::UploadResult* uploadResult) {
    return uploadResult->getConstMetadata(arrayId)->get<arrus::devices::FrameChannelMapping>("frameChannelMapping");
}

std::shared_ptr<arrus::framework::DataBuffer> getFifoLockFreeBuffer(arrus::session::UploadResult* uploadResult) {
    auto buffer = std::static_pointer_cast<DataBuffer>(uploadResult->getBuffer());
    return buffer;
}

// GIL-free methods.
// TODO consider using -threads parameter, or finding a SWIG feature that allows to turn off GIL
// for a particular method (%thread arrus::session::Session::stopScheme() seems to not work).
void arrusSessionStartScheme(std::shared_ptr<arrus::session::Session> session) {
    ArrusPythonGILUnlock unlock;
    session->startScheme();
}

void arrusSessionStopScheme(std::shared_ptr<arrus::session::Session> session) {
    ArrusPythonGILUnlock unlock;
    session->stopScheme();
}


%};
// ------------------------------------------ DEVICES
%{
#include "arrus/core/api/devices/DeviceId.h"
#include "arrus/core/api/devices/Device.h"
#include "arrus/core/api/devices/DeviceWithComponents.h"
#include "arrus/core/api/devices/us4r/Us4OEM.h"
#include "arrus/core/api/devices/us4r/Us4R.h"
#include "arrus/core/api/devices/File.h"
#include "arrus/core/api/devices/probe/ProbeModelId.h"
#include "arrus/core/api/devices/probe/Probe.h"
#include "arrus/core/api/devices/probe/ProbeModel.h"

using namespace arrus::devices;
%};

%ignore operator<<(std::ostream &os, const DeviceId &id);
%include "arrus/core/api/devices/DeviceId.h"
%include "arrus/core/api/devices/Device.h"
%include "arrus/core/api/devices/DeviceWithComponents.h"
%include "arrus/core/api/devices/probe/ProbeModelId.h"
%include "arrus/core/api/devices/probe/ProbeModel.h"
%include "arrus/core/api/devices/probe/Probe.h"
%include "arrus/core/api/devices/us4r/Us4OEM.h"
%include "arrus/core/api/devices/us4r/Us4R.h"
%include "arrus/core/api/devices/File.h"


%inline %{
arrus::devices::Us4R *castToUs4r(arrus::devices::Device *device) {
    auto ptr = dynamic_cast<Us4R*>(device);
    if(!ptr) {
        throw std::runtime_error("Given device is not an us4r handle.");
    }
    return ptr;
}

arrus::devices::File *castToFile(arrus::devices::Device *device) {
    auto ptr = dynamic_cast<File*>(device);
    if(!ptr) {
        throw std::runtime_error("Given device is not a file handle.");
    }
    return ptr;
}
// TODO(pjarosik) remove the bellow functions when possible

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
// GIL free methods

float arrusUs4OEMGetFPGATemperature(::arrus::devices::Us4OEM *us4oem) {
    ArrusPythonGILUnlock unlock;
    return us4oem->getFPGATemperature();
}

float arrusUs4OEMGetUCDTemperature(::arrus::devices::Us4OEM *us4oem) {
    ArrusPythonGILUnlock unlock;
    return us4oem->getUCDTemperature();
}

float arrusUs4OEMGetUCDExternalTemperature(::arrus::devices::Us4OEM *us4oem) {
    ArrusPythonGILUnlock unlock;
    return us4oem->getUCDExternalTemperature();
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
#include "arrus/core/api/ops/us4r/DigitalDownConversion.h"
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
%include "arrus/core/api/ops/us4r/DigitalDownConversion.h"


%include "std_vector.i"
%include "typemaps.i"

namespace std {
%template(TxRxVector) vector<arrus::ops::us4r::TxRx>;
%template(ArrusNdArrayVector) vector<arrus::framework::NdArray>;
};

%inline %{

void TxRxVectorPushBack(std::vector<arrus::ops::us4r::TxRx> &txrxs, arrus::ops::us4r::TxRx &txrx) {
    txrxs.push_back(txrx);
}

void VectorFloatPushBack(std::vector<float> &vector, double value) {
    vector.push_back(float(value));
}

void Arrus2dArrayVectorPushBack(
    std::vector<arrus::framework::NdArray> &arrays,
    size_t nRows, size_t nCols, std::vector<float> values, const std::string &placementName, size_t placementOrdinal,
    const std::string &arrayName
) {
    ::arrus::framework::NdArray::Shape shape = {nRows, nCols};
    ::arrus::devices::DeviceId placement(::arrus::devices::parseToDeviceTypeEnum(placementName), placementOrdinal);
    ::arrus::framework::NdArray array(
        (void*)values.data(),
        shape,
        ::arrus::framework::NdArray::DataType::FLOAT32,
        placement,
        arrayName,
        false // is view => copy
    );
    arrays.push_back(array);
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
