#include "arrus/core/api/framework/NdArray.h"

#include <memory>
#include <utility>

#include "arrus/core/api/common.h"

namespace arrus::framework {

struct NdArray::NdArrayBase {
    NdArrayBase(NdArray::Shape shape, NdArray::DataType dataType, devices::DeviceId placement)
        :shape(std::move(shape)), dataType(dataType), placement(placement) {}

    virtual ~NdArrayBase() = default;

    virtual void* getPtr() = 0;

    NdArray::Shape shape;
    NdArray::DataType dataType;
    ::arrus::devices::DeviceId placement;
};

struct NdArrayImpl : public NdArray::NdArrayBase {
    NdArrayImpl(VoidHandle ptr, NdArray::Shape shape, NdArray::DataType dataType, devices::DeviceId placement)
    : NdArray::NdArrayBase(std::move(shape), dataType, placement), ptr(std::move(ptr)) {}

    virtual void* getPtr() {
        return ptr.get();
    }

    VoidHandle ptr;
};

struct NdArrayView: public NdArray::NdArrayBase {
    NdArrayView(void* ptr, NdArray::Shape shape, NdArray::DataType dataType, devices::DeviceId placement)
    : NdArray::NdArrayBase(std::move(shape), dataType, placement), ptr(ptr) {}

    virtual void* getPtr() {
        return ptr;
    }
    void* ptr;
};

struct DataTypeInfo {
    NdArray::DataType dataType;
    size_t size;
};

size_t getDataTypeSize(NdArray::DataType type) {
    switch(type) {
    case NdArray::DataType::INT16:
        return sizeof(short);
    case NdArray::DataType::INT32:
        return sizeof(int);
    }
}


NdArray::NdArray(VoidHandle ptr, Shape shape, DataType dataType, devices::DeviceId placement)
    : impl(std::make_unique<NdArrayImpl>(std::move(ptr), std::move(shape), dataType, placement)) {}

NdArray::NdArray(std::unique_ptr<NdArrayBase> impl): impl(std::move(impl)) {}

NdArray::~NdArray() = default;

short *NdArray::getInt16() { return (short*)impl->getPtr(); }

void *NdArray::getRaw() { return impl->getPtr(); }

const NdArray::Shape &NdArray::getShape() { return impl->shape; }

NdArray::DataType NdArray::getDataType() const { return impl->dataType; }

const devices::DeviceId &NdArray::getPlacement() const { return impl->placement; }

NdArray NdArray::operator[](int n) const {
    size_t offset = 0; // sum n-1 last elements.
    // Determine offset in number of bytes.
    // This will require conversion from data type to it's size
//    create a view with one dimension less, the same data type, placement, with ptr starting in the ptr + n*offset
    Shape newShape{};
    std::unique_ptr<NdArrayBase> viewImpl = std::make_unique<NdArrayView>((char*)(this->impl->getPtr())+n*offset, newShape,
                                              this->getDataType(), this->getPlacement());
    return NdArray{std::move(viewImpl)};
}

}

