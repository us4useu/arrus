%include stdint.i
%include exception.i
%include std_shared_ptr.i
%include std_string.i
%include std_unordered_set.i
%include std_vector.i
%include std_pair.i
%include std_optional.i

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
#include "arrus/core/api/devices/us4r/HVVoltage.h"
using namespace ::arrus;
%};

%typemap(typecheck) std::optional<float> {
    $1 = PyFloat_Check($input) || $input == Py_None;
}

%typemap(typecheck) std::optional<arrus::devices::Ordinal> {
    $1 = PyInt_Check($input) || $input == Py_None;
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

%typemap(in) std::optional<float> %{
    if($input == Py_None) {
        $1 = std::optional<float>();
    }
    else {
        float value = (float)(PyFloat_AsDouble($input));
        $1 = std::optional<float>(value);
    }
    %}

%typemap(out) std::optional<float> %{
    if($1) {
        $result = PyFloat_FromDouble((double)(*$1));
    }
    else {
        $result = Py_None;
        Py_INCREF(Py_None);
    }
    %}

%typemap(out) std::optional<float>* %{
    if($1 && $1->has_value()) {
        $result = PyFloat_FromDouble($1->value());
    }
    else {
        $result = Py_None;
        Py_INCREF(Py_None);
    }
    %}

%typemap(out) const std::optional<float>& %{
    if($1 && $1->has_value()) {
        $result = PyFloat_FromDouble($1->value());
    }
    else {
        $result = Py_None;
        Py_INCREF(Py_None);
    }
    %}

%typemap(in) std::optional<long long> %{
    if($input == Py_None) {
        $1 = std::optional<long long>();
    }
    else {
        long long value = PyLong_AsLong($input);
        if(value > std::numeric_limits<long long>::max() || value < std::numeric_limits<long long>::min()) {
            std::string errorMsg = "Value '" + std::to_string(value) + "' should be in range: ["
                + std::to_string(std::numeric_limits<long long>::min())
                + ", " + std::to_string(std::numeric_limits<long long>::max()) + "]";
            PyErr_SetString(PyExc_ValueError, errorMsg.c_str());
            return NULL;
        }
        $1 = std::optional<long long>(value);
    }
    %}

%typemap(out) std::optional<long long> %{
    if($1) {
        $result = PyLong_FromLong(*$1);
    }
    else {
        $result = Py_None;
        Py_INCREF(Py_None);
    }
    %}

%include "arrus/core/api/common/types.h"

%feature("valuewrapper", "1");
%include "arrus/core/api/devices/us4r/HVVoltage.h"
%include "arrus/core/api/devices/us4r/HVPSMeasurement.h"
%include "arrus/core/api/devices/GpuSettings.h";
%feature("valuewrapper", "0");

namespace std {
%template(VectorBool) vector<bool>;
%template(VectorFloat) vector<float>;
%template(VectorUInt16) vector<unsigned short>;
%template(VectorUInt8) vector<unsigned char>;
%template(VectorInt8) vector<int8_t>;
%template(VectorInt64) vector<int64_t>;
%template(VectorSizet) vector<size_t>;
%template(PairUint32) pair<unsigned, unsigned>;
%template(PairChannelIdx) pair<unsigned short, unsigned short>;
%template(VectorHVVoltage) vector<arrus::devices::HVVoltage>;
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
#include "arrus/core/api/io/settings.h"
#include "arrus/core/api/session/Session.h"
#include "arrus/core/api/common/logging.h"
#include "arrus/core/api/devices/us4r/Us4OEM.h"
#include "arrus/core/api/devices/us4r/Us4R.h"
#include "arrus/core/api/ops/us4r/TxRxSequence.h"
#include "arrus/core/api/devices/File.h"
#include "arrus/core/api/devices/Gpu.h"

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

%feature("valuewrapper", "1");
%include "arrus/core/api/devices/probe/Lens.h"
%include "arrus/core/api/devices/probe/MatchingLayer.h"
%feature("valuewrapper", "0");


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
        LOGGING_FACTORY->addLogFile(filepath, level);
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
%include "arrus/core/api/common/Slice.h"
%include "arrus/core/api/common/Interval.h"
%include "arrus/core/api/common/Span.h"
%include "arrus/core/api/ops/us4r/DigitalDownConversion.h"
%include "arrus/core/api/common/Parameters.h"

%feature("valuewrapper", "0");

%inline %{
    size_t castToInt(short* ptr) {
        return (size_t)ptr;
    }
    size_t castUint8ToInt(unsigned char* ptr) {
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
    %template(IntervalVoltage) Interval<Voltage>;
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
%shared_ptr(arrus::session::SessionSettings);
%ignore createSession;

// Ignore overloaded `run` methods -- the full signature will be used only.
%ignore arrus::session::Session::run();
%ignore arrus::session::Session::run(bool);


%include "arrus/core/api/session/Metadata.h"
%include "arrus/core/api/session/UploadResult.h"
%include "arrus/core/api/session/Session.h"

%inline %{

std::shared_ptr<arrus::session::Session> createSessionSharedHandle(const std::string& filepath) {
    std::shared_ptr<Session> res = createSession(filepath);
    return res;
}

float getRxOffset(size_t arrayId, arrus::session::UploadResult* uploadResult) {
    return *uploadResult->getConstMetadata(arrayId)->get<float>("rxOffset");
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


void arrusSessionClose(std::shared_ptr<arrus::session::Session> session) {
    ArrusPythonGILUnlock unlock;
    session->close();
}


void arrusSessionRun(std::shared_ptr<arrus::session::Session> session, bool async, std::optional<long long> timeout) {
    ArrusPythonGILUnlock unlock;
    session->run(async, timeout);
}

void arrusUs4OEMWaitForHVPSMeasuerementDone(arrus::devices::Us4OEM *us4oem, std::optional<long long> timeout) {
    ArrusPythonGILUnlock unlock;
    us4oem->waitForHVPSMeasurementDone(timeout);
}

void arrusUs4RSetVoltage(arrus::devices::Us4R *us4r, float voltage) {
    ArrusPythonGILUnlock unlock;
    us4r->setVoltage(voltage);
}

void arrusUs4RSetVoltageMulti(arrus::devices::Us4R *us4r,
                              const std::vector<arrus::devices::HVVoltage> &voltages) {
    ArrusPythonGILUnlock unlock;
    us4r->setVoltage(voltages);
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
#include  "arrus/core/api/devices/Gpu.h"
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
%include "arrus/core/api/devices/probe/ProbeModelId.h"
%include "arrus/core/api/devices/probe/ProbeModel.h"
%include "arrus/core/api/devices/probe/Probe.h"
%include "arrus/core/api/devices/Gpu.h"

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

arrus::devices::Gpu *castToGpu(arrus::devices::Device *device) {
    auto ptr = dynamic_cast<Gpu*>(device);
    if(!ptr) {
        throw std::runtime_error("Given device is not a GPU handle.");
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
#include "arrus/core/api/ops/us4r/Waveform.h"
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
%ignore arrus::ops::us4r::Pulse::toWaveform() const;
%include "arrus/core/api/ops/us4r/Pulse.h"
%include "arrus/core/api/ops/us4r/Waveform.h"
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
%template(SliceVector) vector<arrus::Slice>;
%template(OptionalFloatVector) vector<std::optional<float>>;
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

void SlicePushBack(std::vector<arrus::Slice> &vector, arrus::Slice &slice) {
    vector.push_back(slice);
}

void OptionalVectorFloatPushBack(std::vector<std::optional<float>> &vector, std::optional<float> value) {
    vector.push_back(value);
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
#include "arrus/core/api/devices/GpuSettings.h"
#include "arrus/core/api/devices/us4r/WatchdogSettings.h"
#include "arrus/core/api/devices/us4r/RxSettings.h"
#include "Us4OEM/api/RxSettings.h"
#include "arrus/core/api/devices/us4r/Us4OEMSettings.h"
#include "arrus/core/api/devices/us4r/ProbeAdapterSettings.h"
#include "arrus/core/api/devices/us4r/ProbeAdapterModelId.h"
#include "arrus/core/api/devices/probe/ProbeSettings.h"
#include "arrus/core/api/devices/probe/ProbeModel.h"
#include "arrus/core/api/devices/probe/ProbeModelId.h"
#include "arrus/core/api/devices/us4r/HVSettings.h"
#include "arrus/core/api/devices/us4r/HVModelId.h"
#include "arrus/core/api/devices/us4r/Us4OEMInterrupt.h"
#include "arrus/core/api/devices/us4r/Us4RSettings.h"
#include "arrus/core/api/session/SessionSettings.h"

using namespace ::arrus::devices;
using namespace ::us4us::us4r;
%};

%include "arrus/core/api/devices/us4r/WatchdogSettings.h";
%include "arrus/core/api/devices/us4r/RxSettings.h";
%include "Us4OEM/api/RxSettings.h"
%include "arrus/core/api/devices/us4r/Us4OEMSettings.h";
%ignore operator<<(std::ostream &os, const ProbeAdapterModelId &id);
%include "arrus/core/api/devices/us4r/ProbeAdapterModelId.h";
%include "arrus/core/api/devices/us4r/ProbeAdapterSettings.h";
%ignore operator<<(std::ostream &os, const ProbeModelId &id);
%include "arrus/core/api/devices/probe/ProbeModelId.h";
%include "arrus/core/api/devices/probe/Lens.h";
%include "arrus/core/api/devices/probe/MatchingLayer.h";
%include "arrus/core/api/devices/probe/ProbeModel.h";
%include "arrus/core/api/devices/probe/ProbeSettings.h"
%ignore operator<<(std::ostream &os, const HVModelId &id);
%include "arrus/core/api/devices/us4r/HVModelId.h"
%include "arrus/core/api/devices/us4r/HVSettings.h"
%include "arrus/core/api/devices/us4r/Us4OEMInterrupt.h"
%include "arrus/core/api/devices/us4r/Us4RSettings.h"
%include "arrus/core/api/session/SessionSettings.h"

// ------------------------------------------ IO
%include "arrus/core/api/io/settings.h"

// ------------------------------------------ SYSTEM INTERRUPT CALLBACK
%feature("director") Us4OEMInterruptListener;

%inline %{

/**
 * Director-style listener Python users subclass to receive us4OEM system
 * interrupts. Override one of the per-interrupt methods (on_probe_not_connected,
 * on_pulser_interrupt, on_tx_timeout, on_watchdog0, on_watchdog1) to handle a
 * single interrupt, or override on_any_interrupt to handle every supported
 * interrupt with one function. If both kinds of overrides are provided, the
 * specific per-interrupt override wins for that interrupt.
 *
 * The runtime calls these methods from a us4r-api interrupt thread; the C++
 * adapter acquires the GIL before invoking them.
 *
 * The listener instance must remain alive for as long as the Session that
 * was created from the SessionSettings holding the adapted callbacks.
 */
class Us4OEMInterruptListener {
public:
    Us4OEMInterruptListener() = default;
    virtual ~Us4OEMInterruptListener() = default;

    /**
     * Catch-all hook invoked by the default per-interrupt methods.
     * Override to handle every supported interrupt with a single function.
     */
    virtual void on_any_interrupt(arrus::devices::Us4OEMInterrupt /*interrupt*/,
                                  arrus::devices::Ordinal /*oem*/) {}

    virtual void on_probe_not_connected(arrus::devices::Ordinal oem) {
        on_any_interrupt(arrus::devices::Us4OEMInterrupt::PROBE_NOT_CONNECTED, oem);
    }
    virtual void on_pulser_interrupt(arrus::devices::Ordinal oem) {
        on_any_interrupt(arrus::devices::Us4OEMInterrupt::PULSER_INTERRUPT, oem);
    }
    virtual void on_tx_timeout(arrus::devices::Ordinal oem) {
        on_any_interrupt(arrus::devices::Us4OEMInterrupt::TX_TIMEOUT, oem);
    }
    virtual void on_watchdog0(arrus::devices::Ordinal oem) {
        on_any_interrupt(arrus::devices::Us4OEMInterrupt::WATCHDOG_IRQ0, oem);
    }
    virtual void on_watchdog1(arrus::devices::Ordinal oem) {
        on_any_interrupt(arrus::devices::Us4OEMInterrupt::WATCHDOG_IRQ1, oem);
    }
    virtual void on_hvps_fuse(arrus::devices::Ordinal oem) {
        on_any_interrupt(arrus::devices::Us4OEMInterrupt::HVPS_FUSE, oem);
    }
};

%};

%{
namespace {

template <class F>
arrus::devices::Us4OEMInterruptCallback
makeListenerCallback(Us4OEMInterruptListener *listener, F method) {
    return [listener, method](arrus::devices::Ordinal oem) {
        PyGILState_STATE gstate = PyGILState_Ensure();
        try {
            (listener->*method)(oem);
        } catch (const std::exception &e) {
            std::cerr << "Exception in interrupt listener: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Unhandled exception in interrupt listener" << std::endl;
        }
        PyGILState_Release(gstate);
    };
}

}// namespace
%}

%inline %{

/**
 * Loads SessionSettings from a prototxt configuration file and attaches the
 * given listener as the us4OEM interrupt callbacks for every Us4R in the
 * configuration. One callback is registered per supported interrupt; each
 * dispatches to the matching listener method. Returns an opaque shared
 * handle that can be passed to createSessionSharedHandleFromSettings.
 */
std::shared_ptr<arrus::session::SessionSettings>
createSessionSettingsFrom(const std::string &filepath,
                          Us4OEMInterruptListener &listener) {
    auto loaded = arrus::io::readSessionSettings(filepath);

    Us4OEMInterruptListener *listenerPtr = &listener;
    arrus::devices::Us4OEMInterruptCallbacksMap cbs;
    cbs[arrus::devices::Us4OEMInterrupt::PROBE_NOT_CONNECTED] =
        makeListenerCallback(listenerPtr, &Us4OEMInterruptListener::on_probe_not_connected);
    cbs[arrus::devices::Us4OEMInterrupt::PULSER_INTERRUPT] =
        makeListenerCallback(listenerPtr, &Us4OEMInterruptListener::on_pulser_interrupt);
    cbs[arrus::devices::Us4OEMInterrupt::TX_TIMEOUT] =
        makeListenerCallback(listenerPtr, &Us4OEMInterruptListener::on_tx_timeout);
    cbs[arrus::devices::Us4OEMInterrupt::WATCHDOG_IRQ0] =
        makeListenerCallback(listenerPtr, &Us4OEMInterruptListener::on_watchdog0);
    cbs[arrus::devices::Us4OEMInterrupt::WATCHDOG_IRQ1] =
        makeListenerCallback(listenerPtr, &Us4OEMInterruptListener::on_watchdog1);
    cbs[arrus::devices::Us4OEMInterrupt::HVPS_FUSE] =
        makeListenerCallback(listenerPtr, &Us4OEMInterruptListener::on_hvps_fuse);

    arrus::session::SessionSettingsBuilder builder;
    for (const auto &u : loaded.getUs4Rs()) {
        builder.addUs4R(arrus::devices::Us4RSettingsBuilder(u)
                            .setInterruptCallbacks(cbs)
                            .build());
    }
    for (const auto &f : loaded.getFiles()) {
        builder.addFile(f);
    }
    for (size_t i = 0; i < loaded.getNumberOfGpus(); ++i) {
        builder.addGpu(loaded.getGpuSettings(i));
    }
    return std::make_shared<arrus::session::SessionSettings>(builder.build());
}

/**
 * Creates a Session from a previously built SessionSettings handle.
 */
std::shared_ptr<arrus::session::Session>
createSessionSharedHandleFromSettings(
    const std::shared_ptr<arrus::session::SessionSettings> &settings) {
    return std::shared_ptr<arrus::session::Session>(
        arrus::session::createSession(*settings).release());
}

%};
