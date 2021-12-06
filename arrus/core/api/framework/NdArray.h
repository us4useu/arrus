#ifndef ARRUS_CORE_API_FRAMEWORK_ND_ARRAY_H
#define ARRUS_CORE_API_FRAMEWORK_ND_ARRAY_H

#include <utility>
#include <memory>
#include <functional>

#include "arrus/core/api/common/Tuple.h"
#include "arrus/core/api/devices/DeviceId.h"
#include "arrus/core/api/common.h"

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
        INT32,
        FLOAT32,
        FLOAT64,
    };

    /** Array shape. */
    typedef Tuple<unsigned int> Shape;

   /**
    * Creates a new NdArray.
    *
    * The created NdArray becomes an owner of the given data pointed by given pointer.
    *
    * Note: this method moves the ptr to the NdArray internals, so it is no longer valid.
    *
    * @param ptr pointer to data that should be wrapped by this class
    * @param shape shape of the array
    * @param dataType data type of array values
    * @param placement id of the device where the Array is allocated
    */
    static NdArray createArray(VoidHandle ptr, NdArray::Shape shape, NdArray::DataType dataType,
                               devices::DeviceId placement) {
        return NdArray(std::move(ptr), std::move(shape), dataType, placement);
    }

    /**
     * Creates a new NdArray.
     *
     * Note: this method moves the ptr to the NdArray internals, so it is no longer valid.
     *
     * The created NdArray becomes an owner of the given data pointed by given pointer.
     *
     * @param ptr pointer to data that should be wrapped by this class
     * @param shape shape of the array
     * @param dataType data type of array values
     * @param placement id of the device where the Array is allocated
     */
    NdArray(VoidHandle ptr, Shape shape, DataType dataType, devices::DeviceId placement);

    virtual ~NdArray();

    /**
    * Returns a pointer to data.
    *
    * @tparam T data type
    * @return a pointer to data
    */
    template<typename T>
    T *get() {
        return (T *) getRaw();
    }

    /**
     * Returns a pointer to the memory data (assuming the data type is int16).
     * @return pointer to the data array, casted to int16 value.
     */
    short* getInt16();

    /**
     * Returns data shape.
     */
    const Shape &getShape();

    /**
     * Returns array's data type.
     */
    DataType getDataType() const;

    const devices::DeviceId &getPlacement() const;

    /**
     * Returns a view of NdArray for
     * @param n
     * @return
     */
    NdArray operator[](const ::arrus::Tuple<int> ) const;

private:
    struct NdArrayImpl;

    explicit NdArray(std::unique_ptr<NdArrayImpl> impl);
    void *getRaw();

    std::unique_ptr<NdArrayImpl> impl;
};

}

#endif //ARRUS_CORE_API_FRAMEWORK_ND_ARRAY_H
