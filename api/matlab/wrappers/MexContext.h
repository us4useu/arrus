#ifndef ARRUS_API_MATLAB_WRAPPERS_MEXCONTEXT_H
#define ARRUS_API_MATLAB_WRAPPERS_MEXCONTEXT_H

#include <format>
#include <memory>
#include <algorithm>

#include <std4us/string.h>

#include "arrus/core/api/arrus.h"

#include "mex_headers.h"


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
                std::format("Exception while creating object '{}': {}", typeId, e.what()));
        }
    }

    bool isInstance(const ::matlab::data::ObjectArray &object, const std::string& className) {
        try {
            return matlabEngine->feval(u"isa",
             std::vector<::matlab::data::Array>({object, factory.createScalar(className)})
             )[0];
        } catch (const std::exception &e) {
            throw ::arrus::IllegalArgumentException(
                std::format("Exception while calling 'isa': {}", e.what()));
        }
    }

    template<typename T>::matlab::data::TypedArray<T> createScalar(const T &value) {
        try {
            return getArrayFactory().createScalar<T>(value);
        } catch (const std::exception &e) {
            throw ::arrus::IllegalArgumentException(
                std::format("Exception while creating scalar array '{}': {}", value, e.what()));
        }
    }

    ::matlab::data::TypedArray<::matlab::data::MATLABString> createScalarString(const ::matlab::data::MATLABString &v) {
        try {
            return getArrayFactory().createScalar(v);
        } catch (const std::exception &e) {
            throw ::arrus::IllegalArgumentException(
                std::format("Exception while creating scalar string array: {}", e.what()));
        }
    }

    ::matlab::data::TypedArray<::matlab::data::MATLABString> createScalarString(const ::std::string &v) {
        try {
            return getArrayFactory().createScalar(v);
        } catch (const std::exception &e) {
            throw ::arrus::IllegalArgumentException(
                std::format("Exception while creating scalar string array: {}", e.what()));
        }
    }

    template<typename T>::matlab::data::TypedArray<T> createVector(const std::vector<T> &value) {
        try {
            ::matlab::data::ArrayDimensions dimensions = {1, value.size()};
            return getArrayFactory().createArray(dimensions, std::begin(value), std::end(value));
        } catch (const std::exception &e) {
            throw ::arrus::IllegalArgumentException(std::format(
                "Exception while creating vector array '{}': {}", std4us::join(value, ", "), e.what()));
        }
    }

    ::matlab::data::Array createArray(const ::arrus::framework::NdArray &array) {
        try {
            switch(array.getDataType()) {
            case ::arrus::framework::NdArray::DataType::INT16:
                return createTypedArray<::arrus::int16>(array);
            case ::arrus::framework::NdArray::DataType::FLOAT32:
                return createTypedArray<::arrus::float64>(array);
            default:
                throw IllegalArgumentException(std::format("Unhandled arrus data type: {}",
                                               std::to_string(size_t(array.getDataType()))));
            }
        } catch (const std::exception &e) {
            throw IllegalArgumentException(std::format("Exception while creating array: {}", e.what()));
        }
    }

    template<typename T>
    ::matlab::data::Array createTypedArray(const ::arrus::framework::NdArray &array) {
        ::matlab::data::ArrayDimensions dims = array.getShape().getValues();
        // Note: C-contiguous shape to F-shape (just reverse orders).
        std::reverse(std::begin(dims), std::end(dims));
        auto nElements = array.getNumberOfElements();
        auto *start = array.get<T>();
        auto *end = start + nElements;
        return getArrayFactory().createArray(dims, start, end);
    }

    template<typename T>
    ::matlab::data::Array createTypedArray(const std::vector<T> &array, const framework::NdArray::Shape &shape) {
        ::matlab::data::ArrayDimensions dims = shape.getValues();
        // Note: C-contiguous shape to F-shape (just reverse orders).
        std::reverse(std::begin(dims), std::end(dims));
        auto nElements = array.size();
        if(nElements == 0) {
            return getArrayFactory().createEmptyArray();
        }
        auto *start = &array[0];
        auto *end = start + nElements;
        return getArrayFactory().createArray(dims, start, end);
    }

    ::arrus::framework::NdArray createNdArray(const ::matlab::data::Array &array,
                                              const std::string &placement, const std::string &name) {
        try {
            switch(array.getType()) {
            case ::matlab::data::ArrayType::SINGLE:
                return createTypedNdArrayCastFloat(array, placement, name);
            case ::matlab::data::ArrayType::DOUBLE:
                return createTypedNdArrayCastFloat(array, placement, name);
            default:
                throw IllegalArgumentException(std::format("Unhandled arrus data type: {}",
                                                      std::to_string(size_t(array.getType()))));
            }
        } catch (const std::exception &e) {
            throw IllegalArgumentException(std::format("Exception while creating array: {}", e.what()));
        }
    }

    ::arrus::framework::NdArray createTypedNdArrayCastFloat(
        const ::matlab::data::TypedArray<double> &array, const std::string &placement,
        const std::string &name) {

        if(array.isEmpty()) {
            return ::arrus::framework::NdArray();
        }
        ::matlab::data::ArrayDimensions dims = array.getDimensions();
        if(array.getMemoryLayout() == ::matlab::data::MemoryLayout::COLUMN_MAJOR) {
            // Note: F-contiguous shape to C-shape (just reverse orders).
            std::reverse(std::begin(dims), std::end(dims));
        }

        const auto shape = ::arrus::framework::NdArrayDef::Shape(dims);
        const auto p = ::arrus::devices::DeviceId::parse(placement);

        // TODO: we are doing double copy here..., so this is not the most optimal way to upload constants,
        // anyway, should be sufficient for now
        std::vector<float> values(array.getNumberOfElements());
        std::transform(
            std::begin(array), std::end(array), std::begin(values),
            [](const auto v){return (float)v; }
        );
        return ::arrus::framework::NdArray::asarray<float>(values, shape, p, name);
    }

private:
    ::matlab::data::ArrayFactory factory;
    MatlabEnginePtr matlabEngine;
    Logger::SharedHandle defaultLogger;
};

}// namespace arrus::matlab

#endif//ARRUS_API_MATLAB_WRAPPERS_MEXCONTEXT_H
