#include "Model/DataPtr.h"

DataPtr::DataPtr() {
    this->allocatedDataSize = 0;
}

DataPtr::DataPtr(const boost::variant<short *, int *, float *, double *, float2 *> ptr, const Dims dims) : ptr(ptr),
                                                                                                           dims(dims) {
    this->allocatedDataSize = this->getDataSize();
}

DataPtr::~DataPtr() {
}

void DataPtr::setDims(const Dims dims) {
    this->dims = dims;
}

Dims DataPtr::getDims() {
    return this->dims;
}

int DataPtr::getAllocatedDataSize() {
    return this->allocatedDataSize;
}

int DataPtr::getDataSize() {
    return boost::apply_visitor(PtrSizeVisitor(), this->ptr) * this->dims.flatten();
}

boost::variant<short *, int *, float *, double *, float2 *> DataPtr::getRawPtr() {
    return this->ptr;
}

void *DataPtr::getVoidPtr() {
    return *boost::apply_visitor(GenericReferenceToVoidPointerVisitor(), this->ptr);
}

void **DataPtr::getReferenceToVoidPtr() {
    return boost::apply_visitor(GenericReferenceToVoidPointerVisitor(), this->ptr);
}

void DataPtr::copyExtraData(const DataPtr &ptr) {
    this->ptrProperties = ptr.ptrProperties;
}

void DataPtr::setPtrProperty(const propertyName name, const VariableAnyValue value) {
    this->ptrProperties[name] = value;
}

VariableAnyValue &DataPtr::getPtrProperty(const propertyName name) {
    std::unordered_map<propertyName, VariableAnyValue>::iterator it = this->ptrProperties.find(name);
    if(it == this->ptrProperties.end())
        DISPATCHER_LOG(DispatcherLogType::FATAL, std::string("Data pointer doesn't have property named ") + name);
    return it->second;
}

std::unordered_map <propertyName, VariableAnyValue> &DataPtr::getPtrProperties() {
    return this->ptrProperties;
}

void DataPtr::shiftByOffset(const int offset) {
    boost::apply_visitor(GenericPointerOffsetShiftVisitor(offset), this->ptr);
}

void DataPtr::setPtrProperties(const std::unordered_map <propertyName, VariableAnyValue> &properties) {
    this->ptrProperties.insert(properties.begin(), properties.end());
}