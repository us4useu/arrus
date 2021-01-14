#pragma once

#include <boost/variant.hpp>
#include "Model/VariableAnyValue.h"
#include <unordered_map>
#include "Utils/DispatcherLogger.h"
#include <cuda_runtime_api.h>

class GenericReferenceToVoidPointerVisitor : public boost::static_visitor<void **> {
public:
    template<typename T>
    void **operator()(const T &ptr) const {
        return (void **) &ptr;
    }
};

class GenericPointerOffsetShiftVisitor : public boost::static_visitor<> {
private:
    int offset;
public:
    GenericPointerOffsetShiftVisitor(const int offset) : offset(offset) {}

    template<typename T>
    void operator()(T &ptr) const {
        ptr += offset;
    }
};

class PtrSizeVisitor : public boost::static_visitor<int> {
public:
    int operator()(const short *ptr) const {
        return sizeof(short);
    }

    int operator()(const int *ptr) const {
        return sizeof(int);
    }

    int operator()(const float *ptr) const {
        return sizeof(float);
    }

    int operator()(const double *ptr) const {
        return sizeof(double);
    }

    int operator()(const float2 *ptr) const {
        return sizeof(float2);
    }
};

typedef std::string propertyName;

class Dims {
public:
    unsigned int x;
    unsigned int y;
    unsigned int z;

    Dims() : x(0), y(0), z(0) {};

    Dims(const unsigned int x, const unsigned int y = 1, const unsigned int z = 1) : x(x), y(y), z(z) {};

    const unsigned int flatten() const { return this->x * this->y * this->z; }
};

class DataPtr {
private:
    std::unordered_map <propertyName, VariableAnyValue> ptrProperties;
    boost::variant<short *, int *, float *, double *, float2 *> ptr;
    Dims dims;
    int allocatedDataSize;

public:
    DataPtr();

    DataPtr(const boost::variant<short *, int *, float *, double *, float2 *> ptr, const Dims dims);

    ~DataPtr();

    void setDims(const Dims dims);

    Dims getDims();

    int getDataSize();

    int getAllocatedDataSize();

    template<typename T>
    T getPtr() {
        try {
            return boost::get<T>(ptr);
        }
        catch(boost::bad_get exception) {
            DISPATCHER_LOG(DispatcherLogType::ERROR_, std::string("Bad data pointer type cast with generic value."));
            return static_cast<T>(0);
        }
    }

    boost::variant<short *, int *, float *, double *, float2 *> getRawPtr();

    void *getVoidPtr();

    void **getReferenceToVoidPtr();

    // copy everything except pointer variable
    void copyExtraData(const DataPtr &ptr);

    void setPtrProperty(const propertyName name, const VariableAnyValue value);

    void setPtrProperties(const std::unordered_map <propertyName, VariableAnyValue> &properties);

    VariableAnyValue &getPtrProperty(const propertyName name);

    std::unordered_map <propertyName, VariableAnyValue> &getPtrProperties();

    void shiftByOffset(const int offset);
};

