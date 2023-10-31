#ifndef ARRUS_CORE_API_FRAMEWORK_ND_ARRAY_H
#define ARRUS_CORE_API_FRAMEWORK_ND_ARRAY_H

#include <utility>
#include <cstring>
#include <sstream>

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
        : shape(std::move(shape)), dataType(dataType), placement(std::move(placement)),
          isView(false), name(std::move(name)) {

        this->nBytes = this->shape.product() * getDataTypeSize(this->dataType);
        this->ptr = new char[this->nBytes];
        std::memset((char*)(this->ptr), 0, this->nBytes);
    }

    NdArray(void *ptr, Shape shape, DataType dataType, const devices::DeviceId &placement) :
        ptr(ptr), shape(std::move(shape)), dataType(dataType), placement(placement), isView(true) {
        this->nBytes = this->shape.product() * getDataTypeSize(this->dataType);
    }

    NdArray(void *ptr, Shape shape, DataType dataType, const devices::DeviceId &placement, std::string name, bool isView):
        shape(std::move(shape)), dataType(dataType), placement(placement), isView(isView), name(std::move(name)) {
        this->nBytes = this->shape.product() * getDataTypeSize(this->dataType);
        if(isView) {
            this->ptr = ptr;
        } else {
            this->ptr = new char[this->nBytes];
            std::memcpy(this->ptr, ptr, this->nBytes);
        }
    }

    NdArray(const NdArray &other): shape(std::move(other.shape)), dataType(other.dataType), placement(other.placement),
                              isView(other.isView), name(other.name), nBytes(other.nBytes) {
        if(other.isView) {
            this->ptr = other.ptr;
        }
        else {
            this->ptr = new char[this->nBytes];
            std::memcpy(this->ptr, other.ptr, this->nBytes);
        }
    }

    NdArray(NdArray &&other): ptr(other.ptr), shape(std::move(other.shape)), dataType(other.dataType),
                              placement(other.placement), isView(other.isView), name(other.name), nBytes(other.nBytes) {
        other.ptr = nullptr;
        other.nBytes = 0;
    }

    NdArray& operator=(NdArray&& rhs) noexcept {
        if(this == &rhs) {
            return *this;
        }
        this->shape = rhs.shape;
        this->dataType = rhs.dataType;
        this->placement = rhs.placement;
        this->name = rhs.name;

        if(!this->isView) {
            delete (char*)this->ptr;
            this->nBytes = 0;
        }
        this->ptr = rhs.ptr;
        this->nBytes = rhs.nBytes;
        this->isView = rhs.isView;
        rhs.ptr = nullptr;
        rhs.nBytes = 0;

        return *this;
    }

    NdArray& operator=(const NdArray& rhs) noexcept {
        if(this == &rhs) {
            return *this;
        }
        this->shape = rhs.shape;
        this->dataType = rhs.dataType;
        this->placement = rhs.placement;
        this->name = rhs.name;

        if(!this->isView) {
            delete (char*)this->ptr;
            this->nBytes = 0;
        }
        if(!rhs.isView) {
            this->nBytes = rhs.nBytes;
            this->ptr = new char[this->nBytes];
            std::memcpy(this->ptr, rhs.ptr, this->nBytes);
        }
        else {
            this->nBytes = rhs.nBytes;
            this->ptr = rhs.ptr;
        }
        this->isView = rhs.isView;

        return *this;
    }


    NdArray zerosLike() const {
        NdArray array(this->shape, this->dataType, this->placement, this->name);
        return array;
    }

    // TODO implement - copy, move constructors

    virtual ~NdArray() {
        if(!isView && this->ptr != nullptr && this->nBytes > 0) {
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
    T get(size_t row, size_t column) const {
        if(this->shape.size() != 2) {
            throw ::arrus::IllegalArgumentException("The array is expected to be 2D.");
        }
        size_t height = this->shape[0];
        size_t width = this->shape[1];
        if(row >= height || column >= width) {
            throw ::arrus::IllegalArgumentException(
                "Accessing arrays out of bounds, "
                "dimensions: " + std::to_string(height) + ", " + std::to_string(height) +
                "indices: " + std::to_string(row) + ", " + std::to_string(column));
        }
        T* dst = (T*)ptr + (row*width + column);
        return *dst;
    }

    template<typename T>
    void set(size_t row, size_t column, T value) {
        if(this->shape.size() != 2) {
            throw ::arrus::IllegalArgumentException("The array is expected to be 2D.");
        }
        size_t height = this->shape[0];
        size_t width = this-> shape[1];
        if(row >= height || column >= width) {
            throw ::arrus::IllegalArgumentException(
                "Accessing arrays out of bounds, "
                "dimensions: " + std::to_string(height) + ", " + std::to_string(height) +
                    "indices: " + std::to_string(row) + ", " + std::to_string(column));
        }
        T* dst = (T*)ptr + (row*width + column);
        *dst = value;
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

    const std::string toString() const {
	if(this->shape.size() != 2 || this->dataType != NdArray::DataType::FLOAT32) {
		throw ::arrus::IllegalArgumentException("Currently toString supports 2D float32 arrays only.");
	}
	std::stringstream ss;
	for(size_t r = 0; r < this->shape[0]; ++r) {
		for(size_t c = 0; c < this->shape[1]; ++c) {
			ss << this->get<float>(r, c) << ", ";
		}
		ss << std::endl;
	}
	return ss.str();
    }

private:
    void *ptr;
    Shape shape;
    DataType dataType;
    ::arrus::devices::DeviceId placement;
    bool isView;
    std::string name{};
    size_t nBytes;
};

}

#endif //ARRUS_CORE_API_FRAMEWORK_ND_ARRAY_H
