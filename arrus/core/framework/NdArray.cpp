#include "arrus/core/api/framework/NdArray.h"

#include <memory>
#include <utility>

#include "arrus/core/api/common.h"

namespace arrus::framework {

struct NdArray::NdArrayImpl {
    NdArrayImpl(VoidHandle ptr, NdArray::Shape shape, NdArray::DataType dataType, devices::DeviceId placement)
        : ptr(std::move(ptr)), shape(std::move(shape)), dataType(dataType), placement(placement) {}

    virtual ~NdArrayImpl() = default;

    VoidHandle ptr;
    NdArray::Shape shape;
    NdArray::DataType dataType;
    ::arrus::devices::DeviceId placement;
};


NdArray::NdArray(VoidHandle ptr, Shape shape, DataType dataType, devices::DeviceId placement)
    : impl(std::make_unique<NdArrayImpl>(std::move(ptr), std::move(shape), dataType, placement)) {}

NdArray::NdArray(std::unique_ptr<NdArrayImpl> impl): impl(std::move(impl)) {}

NdArray::~NdArray() = default;

short *NdArray::getInt16() { return (short*)impl->ptr.get(); }

void *NdArray::getRaw() { return impl->ptr.get(); }

const NdArray::Shape &NdArray::getShape() { return impl->shape; }

NdArray::DataType NdArray::getDataType() const { return impl->dataType; }

const devices::DeviceId &NdArray::getPlacement() const { return impl->placement; }

NdArray NdArray::operator[](int n) const {
    return NdArray(std::unique_ptr(), arrus::Tuple(), NdArray::DataType::INT16, devices::DeviceId());
}

}

