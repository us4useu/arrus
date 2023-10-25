#ifndef ARRUS_CORE_API_FRAMEWORK_ND_ARRAY_H
#define ARRUS_CORE_API_FRAMEWORK_ND_ARRAY_H

#include <utility>

#include "arrus/core/api/common/Tuple.h"
#include "arrus/core/api/devices/DeviceId.h"

namespace arrus::framework {


/**
 * N-dimensional array.
 *
 * The data order in memory is C-contiguous (last axis varies the fastest).
 *
 * The address returned by `getData` function is located on a device determined by placement property.
 * CPU:0 placement means that the data is located in host computer's RAM.
 *
 */
class NdArray {
public:
    /** A list of currently supported data types of the output buffer.*/
    enum class DataType {
        INT16,
        FLOAT32
    };

    static size_t getDataTypeSize(DataType type) {
        switch(type) {
        case DataType::INT16:
            return sizeof(int16_t);
        case DataType::FLOAT32:
            return sizeof(float32);
        default:
            throw arrus::IllegalArgumentException("Unsupported data type");
        }
    }

    /** Array shape. */
    typedef Tuple<size_t> Shape;

    NdArray(): ptr(nullptr), placement(devices::DeviceId(devices::DeviceType::CPU, 0)){}

    NdArray(Shape shape, DataType dataType, devices::DeviceId placement, std::string name)
        : shape(std::move(shape)), dataType(dataType), placement(std::move(placement)), name(std::move(name)),
          isView(false) {

        size_t bytesToAllocate = shape.product() * getDataTypeSize(dataType);
        ptr = new char[bytesToAllocate];
    }

    NdArray(void *ptr, Shape shape, DataType dataType, const devices::DeviceId &placement) :
        ptr(ptr), shape(std::move(shape)), dataType(dataType), placement(placement), isView(true) {}

    virtual ~NdArray() {
        if(!isView) {
            // NOTE: migration to new framework API: the non-view ndarrays will have the char* ptr property.
            delete[] (char*)ptr;
        }
    }

    /**
    * Returns a pointer to data.
    *
    * @tparam T data type
    * @return a pointer to data
    */
    template<typename T>
    T *get() {
        return (T *) ptr;
    }

    /**
    * Returns a pointer to data.
    *
    * @tparam T data type
    * @return a pointer to data
    */
    template<typename T>
    const T *get() const {
        return (T *) ptr;
    }

    template<typename T>
    void set(size_t i, T value) {
        if(!this->isView) {
            throw ::arrus::IllegalArgumentException("The NdArray value setter can be used only for non-view NdArrays.");
        }
        T* dst = (char*)ptr + i*sizeof(T);
        *dst = value;
    }

    /**
     * Returns a pointer to the memory data (assuming the data type is int16).
     * @return
     */
    short* getInt16() {
        return this->get<short>();
    }

    /**
     * Returns data shape.
     */
    const Shape &getShape() const {
        return shape;
    }

    size_t getNumberOfElements() const {
        return shape.product();
    }

    /**
     * Returns array data type.
     */
    DataType getDataType() const {
        return dataType;
    }

    NdArray view() const {
        return NdArray{ptr, shape, dataType, placement};
    }

    NdArray slice(size_t i, int begin, int end) {
        size_t multiplier = 1;
        for(size_t j = shape.size()-1; j > i; --j) {
            multiplier *= shape[j];
        }
        if(end == -1) {
            end = (int)shape[i];
        }
        Shape newShape = shape.set(i, end-begin);
        return NdArray{((int16_t*)ptr) + multiplier*begin, newShape, dataType, placement};
    }

    const devices::DeviceId &getPlacement() const { return placement; }

    const std::string &getName() const { return name; }

private:
    void *ptr;
    Shape shape;
    DataType dataType;
    ::arrus::devices::DeviceId placement;
    bool isView;
    std::string name{};
};

}

#endif //ARRUS_CORE_API_FRAMEWORK_ND_ARRAY_H
