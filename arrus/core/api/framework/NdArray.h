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
 */
class NdArray {
public:
    /** A list of currently supported data types of the output buffer.*/
    enum class DataType {
        INT16
    };

    /** Array shape. */
    typedef Tuple<size_t> Shape;

    NdArray(void *ptr, Shape shape, DataType dataType, const devices::DeviceId &placement) :
        ptr(ptr), shape(std::move(shape)), dataType(dataType), placement(placement) {}

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

private:
    void *ptr;
    Shape shape;
    DataType dataType;
    ::arrus::devices::DeviceId placement;
};

}

#endif //ARRUS_CORE_API_FRAMEWORK_ND_ARRAY_H
