#ifndef CPP_EXAMPLE_NDARRAY_H
#define CPP_EXAMPLE_NDARRAY_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>
#include <vector>
#include <limits>
#include <sstream>

#include "CudaUtils.cuh"
#include "DataType.h"

namespace arrus_example_imaging {

typedef std::vector<unsigned> DataShape;
typedef DataType DataType;



class NdArrayDef {
public:
    NdArrayDef() = default;

    NdArrayDef(DataShape Shape, DataType Type) : shape(std::move(Shape)), type(Type) {}

    const DataShape &getShape() const { return shape; }
    DataType getType() const { return type; }

private:
    DataShape shape;
    DataType type;
};

class NdArray {
public:

#define ASSERT_NOT_GPU() assertNotGpu(__LINE__)

    /**
     * Creates a scalar NdArray.
     *
     * @param value value to store in NdArray
     * @return scalar NdArray
     */
    template<typename T>
    static NdArray asarray(const T value, bool isGpu = false) {
        std::vector<T> values = {value};
        return NdArray::asarray(values, isGpu);
    }

    static NdArray asarray(const std::vector<double> &values, bool isGpu = false) {
        NdArrayDef def{{(unsigned)values.size(), }, DataType::FLOAT64};
        return NdArray::asarray(def, values, isGpu);
    }

    static NdArray asarray(const std::vector<float> &values, bool isGpu = false) {
        NdArrayDef def{{(unsigned)values.size(), }, DataType::FLOAT32};
        return NdArray::asarray(def, values, isGpu);
    }

    static NdArray asarray(const std::vector<int8_t> &values, bool isGpu = false) {
        NdArrayDef def{{(unsigned)values.size(), }, DataType::INT8};
        return NdArray::asarray(def, values, isGpu);
    }

    static NdArray asarray(const std::vector<uint8_t> &values, bool isGpu = false) {
        NdArrayDef def{{(unsigned)values.size(), }, DataType::UINT8};
        return NdArray::asarray(def, values, isGpu);
    }

    static NdArray asarray(const std::vector<uint16_t> &values, bool isGpu = false) {
        NdArrayDef def{{(unsigned)values.size(), }, DataType::UINT16};
        return NdArray::asarray(def, values, isGpu);
    }

    static NdArray asarray(const std::vector<int16_t> &values, bool isGpu = false) {
        NdArrayDef def{{(unsigned)values.size(), }, DataType::INT16};
        return NdArray::asarray(def, values, isGpu);
    }

    static NdArray asarray(const std::vector<int32_t> &values, bool isGpu = false) {
        NdArrayDef def{{(unsigned)values.size(), }, DataType::INT32};
        return NdArray::asarray(def, values, isGpu);
    }

    static NdArray asarray(const std::vector<uint32_t> &values, bool isGpu = false) {
        NdArrayDef def{{(unsigned)values.size(), }, DataType::UINT32};
        return NdArray::asarray(def, values, isGpu);
    }

    template<typename T>
    static NdArray asarray(NdArrayDef def, std::vector<T> values, bool isGpu = false) {
        NdArray result{def, isGpu};
        cudaMemcpyKind memcpyKind = isGpu ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
        NdArray::memcpy(result.ptr, (uint8_t*)values.data(), result.nBytes, memcpyKind);
        return result;
    }

    template<typename T>
    static NdArray zeros(size_t nElements) {
        std::vector<T> values(nElements, 0);
        return asarray(values);
    }

    template<typename T>
    static NdArray vector(size_t nElements, T value) {
        std::vector<T> values(nElements, value);
        return asarray(values);
    }

    /**
     * Generates a list of values from range [start, stop), with given step value.
     */
    static NdArray arange(float start, float stop, float step = 1.0f) {
        std::vector<float> values;
        if(start >= stop) {
            throw std::runtime_error("NdArray: Start value should be less than stop value: "
                                     + std::to_string(start) + ", "
                                     + std::to_string(stop));
        }
        float current = start;
        while(current < stop) {
            values.push_back(current);
            current += step;
        }
        return asarray(values);
    }

    NdArray() = default;

    NdArray(const NdArrayDef &definition, bool isGpu)
        : ptr(nullptr), shape(definition.getShape()), dataType(definition.getType()), gpu(isGpu) {
        if (shape.empty()) {
            // empty array shape (0)
            return;
        }
        nBytes = calculateSize(shape, dataType);
        allocateMemory(&(this->ptr), nBytes, isGpu);
    }


    NdArray(void *ptr, const NdArrayDef& definition, bool isGpu)
        : ptr((uint8_t *) ptr), shape(definition.getShape()), dataType(definition.getType()), gpu(isGpu) {
        nBytes = calculateSize(shape, dataType);
        isExternal = true;
    }

    NdArray(NdArray &&array) noexcept
        : ptr(array.ptr), shape(std::move(array.shape)), dataType(array.dataType), gpu(array.gpu),
          nBytes(array.nBytes), isExternal(array.isExternal) {
        array.ptr = nullptr;
        array.nBytes = 0;
    }

    NdArray &operator=(NdArray &&array) noexcept {
        if (this != &array) {
            freeMemory();

            ptr = array.ptr;
            array.ptr = nullptr;

            nBytes = array.nBytes;
            array.nBytes = 0;

            shape = std::move(array.shape);
            dataType = array.dataType;
            gpu = array.gpu;
            isExternal = array.isExternal;
        }
        return *this;
    }

    NdArray(const NdArray &array) {
        shape = array.shape;
        dataType = array.dataType;
        nBytes = array.nBytes;
        gpu = array.gpu;
        isExternal = array.isExternal;
        if(! isExternal) {
            allocateMemory(&(this->ptr), nBytes, gpu);
            NdArray::memcpy(this->ptr, array.ptr, nBytes, gpu);
        }
        else {
            ptr = array.ptr;
        }
    }

    NdArray &operator=(const NdArray &array) {
        if (this != &array) {
            freeMemory();
            shape = array.shape;
            dataType = array.dataType;
            nBytes = array.nBytes;
            gpu = array.gpu;
            isExternal = array.isExternal;
            if(! isExternal) {
                allocateMemory(&(this->ptr), nBytes, gpu);
                NdArray::memcpy(this->ptr, array.ptr, nBytes, gpu);
            }
            else {
                ptr = array.ptr;
            }
        }
        return *this;
    }

    virtual ~NdArray() { freeMemory(); }

    template<typename T> T *getPtr() { return (T *) ptr; }

    template<typename T> const T *getConstPtr() const { return (T *) ptr; }

    const std::vector<unsigned> &getShape() const { return shape; }

    DataType getDataType() const { return dataType; }

    NdArrayDef getDef() const { return NdArrayDef{shape, dataType};}

    size_t getNBytes() const { return nBytes; }

    bool isGpu() const { return gpu; }

    void freeMemory() {
        if (ptr == nullptr) {
            return;
        }
        if(isExternal) {
            // external data (views) are not managed by this class
            return;
        }
        if (gpu) {
            CUDA_ASSERT_NO_THROW(cudaFree(ptr));
        } else {
            CUDA_ASSERT_NO_THROW(cudaFreeHost(ptr));
        }
    }

    NdArray createView() {
        return NdArray{ptr, NdArrayDef{shape, dataType}, gpu};
    }

    bool isView() const {
        return isExternal;
    }

    /**
     * Element-wise multiplication by scalar value.
     *
     * @param value value by which this array should multiplied.
     * @return the result array (currently: new array with data type float32)
     */
    NdArray operator*(const float value) const {
        ASSERT_NOT_GPU();
        NdArrayDef resultDef {this->getShape(), DataType::FLOAT32};
        NdArray result{resultDef, this->gpu}; // New, complete array.
        auto nElements = result.getNumberOfElements();
        auto* outputContainer = (float*)result.ptr;

        for(size_t i = 0; i < nElements; ++i) {
            outputContainer[i] = get<float>(i) * value;
        }
        return result;
    }

    /**
     * Element-wise addition with scalar value.
     *
     * @param value value by which this array should multiplied.
     * @return the result array (currently: new array with data type float32)
     */
    NdArray operator+(const float value) const {
        ASSERT_NOT_GPU();
        NdArrayDef resultDef {this->getShape(), DataType::FLOAT32};
        NdArray result{resultDef, this->gpu}; // New, complete array.
        auto nElements = result.getNumberOfElements();
        auto* outputContainer = (float*)result.ptr;

        for(size_t i = 0; i < nElements; ++i) {
            outputContainer[i] = get<float>(i) + value;
        }
        return result;
    }

    /**
     * Element-wise division by scalar value.
     *
     * @param value value by which this array should multiplied.
     * @return the result array (currently: new array with data type float32)
     */
    NdArray operator/(const float value) const {
        ASSERT_NOT_GPU();
        NdArrayDef resultDef {this->getShape(), DataType::FLOAT32};
        NdArray result{resultDef, this->gpu}; // New, complete array.
        auto nElements = result.getNumberOfElements();
        auto* outputContainer = (float*)result.ptr;

        for(size_t i = 0; i < nElements; ++i) {
            outputContainer[i] = get<float>(i) / value;
        }
        return result;
    }

    /**
     * Element-wise addition.
     *
     * Note: currently, the output Array will have float32 data type,
     * regardless of input data type.
     *
     * @return the result array (currently: new array with data type float32)
     */
    NdArray operator+(const NdArray& other) const {
        ASSERT_NOT_GPU();
        auto nElements = this->getNumberOfElements();
        if(nElements != other.getNumberOfElements()) {
            throw std::runtime_error("Both NdArray should have the same size while adding them together");
        }
        NdArrayDef resultDef {this->getShape(), DataType::FLOAT32};
        NdArray result{resultDef, this->gpu}; // New, complete array.
        auto* outputContainer = (float*)result.ptr;

        for(size_t i = 0; i < nElements; ++i) {
            outputContainer[i] = this->get<float>(i) + other.get<float>(i);
        }
        return result;
    }

    /**
     * Element-wise negative value.
     *
     * Note: currently, the output Array will have float32 data type,
     * regardless of input data type.
     *
     * @return the result array (currently: new array with data type float32)
     */
    NdArray operator-() const {
        ASSERT_NOT_GPU();
        NdArrayDef resultDef {this->getShape(), DataType::FLOAT32};
        NdArray result{resultDef, this->gpu};
        auto* outputContainer = (float*)result.ptr;
        auto nElements = this->getNumberOfElements();

        for(size_t i = 0; i < nElements; ++i) {
            outputContainer[i] = -this->get<float>(i);
        }
        return result;
    }

    /**
     * Compute sine of each element this array.
     *
     * Note: currently, the output Array will have float32 data type,
     * regardless of input data type.
     *
     * @return the result array (currently: new array with data type float32)
     */
    NdArray sin() const {
        ASSERT_NOT_GPU();
        NdArrayDef resultDef {this->getShape(), DataType::FLOAT32};
        NdArray result{resultDef, this->gpu};
        auto* outputContainer = (float*)result.ptr;
        auto nElements = this->getNumberOfElements();

        for(size_t i = 0; i < nElements; ++i) {
            outputContainer[i] = std::sin(this->get<float>(i));
        }
        return result;
    }

    /**
     * Compute cosine of each element this array.
     *
     * Note: currently, the output Array will have float32 data type,
     * regardless of input data type.
     *
     * @return the result array (currently: new array with data type float32)
     */
    NdArray cos() const {
        ASSERT_NOT_GPU();
        NdArrayDef resultDef {this->getShape(), DataType::FLOAT32};
        NdArray result{resultDef, this->gpu};
        auto* outputContainer = (float*)result.ptr;
        auto nElements = this->getNumberOfElements();

        for(size_t i = 0; i < nElements; ++i) {
            outputContainer[i] = std::cos(this->get<float>(i));
        }
        return result;
    }

    /**
     * Compute tangent of each element this array.
     *
     * Note: currently, the output Array will have float32 data type,
     * regardless of input data type.
     *
     * @return the result array (currently: new array with data type float32)
     */
    NdArray tang() const {
        ASSERT_NOT_GPU();
        NdArrayDef resultDef {this->getShape(), DataType::FLOAT32};
        NdArray result{resultDef, this->gpu};
        auto* outputContainer = (float*)result.ptr;
        auto nElements = this->getNumberOfElements();

        for(size_t i = 0; i < nElements; ++i) {
            outputContainer[i] = std::tan(this->get<float>(i));
        }
        return result;
    }

    /**
     * Element-wise subtraction.
     *
     * Note: currently, the output Array will have float32 data type,
     * regardless of input data type.
     *
     * @return the result array (currently: new array with data type float32)
     */
    NdArray operator-(const NdArray& other) const {
        return *this + (-other);
    }

    /**
     * Element-wise subtraction.
     *
     * Note: currently, the output Array will have float32 data type,
     * regardless of input data type.
     *
     * @return the result array (currently: new array with data type float32)
     */
    NdArray operator-(const float value) const {
        ASSERT_NOT_GPU();
        NdArrayDef resultDef {this->getShape(), DataType::FLOAT32};
        NdArray result{resultDef, this->gpu};
        auto* outputContainer = (float*)result.ptr;
        auto nElements = this->getNumberOfElements();

        for(size_t i = 0; i < nElements; ++i) {
            outputContainer[i] = this->get<float>(i)-value;
        }
        return result;
    }

    template<typename T>
    T get(size_t i) const {
        ASSERT_NOT_GPU();
        auto nElements = getNumberOfElements();
        if(i >= nElements) {
            throw std::runtime_error("Index " + std::to_string(i) +
                                     " out of array bounds (max: " + std::to_string(nElements) + ")");
        }
        uint8_t* position = ptr+(i* getSizeofDataType(this->dataType));
        return this->castToType<T>(position);
    }

    template<typename T>
    T max() {
        T currentMax = std::numeric_limits<T>::min();
        for(size_t i = 0; i < getNumberOfElements(); ++i) {
            T value = this->get<T>(i);
            if(currentMax < value) {
                currentMax = value;
            }
        }
        return currentMax;
    }

    template<typename T>
    T min() {
        T currentMin = std::numeric_limits<T>::max();
        for(size_t i = 0; i < getNumberOfElements(); ++i) {
            T value = this->get<float>(i);
            if(currentMin > value) {
                currentMin = value;
            }
        }
        return currentMin;
    }

    template<typename T>
    std::vector<T> toVector() {
        auto* data = (T*)this->ptr;
        auto nElements = getNumberOfElements();
        std::vector<T> result(nElements);
        for(size_t i = 0; i < nElements; ++i) {
            result[i] = get<T>(i);
        }
        return result;
    }

    size_t getNumberOfElements() const {
        return getNumberOfElements(this->shape);
    }

    NdArray& reshape(const DataShape& newShape) {
        if(getNumberOfElements() != getNumberOfElements(newShape)) {
            throw std::runtime_error("NdArray: new shape incompatible with the size of array.");
        }
        this->shape = newShape;
        return *this;
    }

    std::string toString() const {
        std::ostringstream ss;
        ss << "NdArray, shape: (";
        for(auto value: shape) {
            ss << value << ",";
        }
        ss << "), values: ";
        for(int i = 0; i < getNumberOfElements(); ++i) {
            ss << get<float>(i) << ", ";
        }
        return ss.str();
    }

    NdArray toGpu() {
        if(gpu) {
            throw std::runtime_error("NdArray already on GPU.");
        }
        NdArray result{NdArrayDef{this->shape, this->dataType}, true};
        memcpy(result.ptr, this->ptr, this->nBytes, cudaMemcpyHostToDevice);
        return result;
    }

private:
    static size_t getSizeofDataType(DataType type){
        if (type == DataType::UINT8) {
            return sizeof(unsigned char);
        } else if (type == DataType::INT8) {
            return sizeof(char);
        } else if (type == DataType::UINT16) {
            return sizeof(unsigned short);
        } else if (type == DataType::INT16) {
            return sizeof(short);
        } if (type == DataType::UINT32) {
            return sizeof(unsigned int);
        } else if (type == DataType::INT32) {
            return sizeof(int);
        } else if (type == DataType::FLOAT32) {
            return sizeof(float);
        } else if (type == DataType::FLOAT64) {
            return sizeof(double);
        } else if (type == DataType::COMPLEX64) {
            return sizeof(float) * 2;
        } else if (type == DataType::COMPLEX128) {
            return sizeof(double) * 2;
        }
        throw std::runtime_error("Unhandled data type");
    }

    template<typename T>
    T castToType(const uint8_t* data) const {
        switch(dataType) {
        case DataType::UINT8:
            return castSrcTypeToDstType<T, unsigned char>(data);
        case DataType::INT8:
            return castSrcTypeToDstType<T, char>(data);
        case DataType::UINT16:
            return castSrcTypeToDstType<T, unsigned short>(data);
        case DataType::INT16:
            return castSrcTypeToDstType<T, short>(data);
        case DataType::UINT32:
            return castSrcTypeToDstType<T, unsigned int>(data);
        case DataType::INT32:
            return castSrcTypeToDstType<T, int>(data);
        case DataType::FLOAT32:
            return castSrcTypeToDstType<T, float>(data);
        case DataType::FLOAT64:
            return castSrcTypeToDstType<T, double>(data);
        default:
            throw std::runtime_error("Cast to type: unhandled data type");
        }
    }

    template<typename Dst, typename Src>
    Dst castSrcTypeToDstType(const uint8_t* data) const {
        return (Dst)(*(Src*)data);
    }

    static size_t calculateSize(const DataShape &shape, DataType type) {
        size_t result = 1;
        for (auto &val : shape) {
            result *= val;
        }
        return result * getSizeofDataType(type);
    }

    static void allocateMemory(uint8_t** dst, size_t size, bool gpu) {
        if (gpu) {
            CUDA_ASSERT(cudaMalloc(dst, size));
        } else {
            CUDA_ASSERT(cudaMallocHost(dst, size));
        }
    }

    static void memcpy(uint8_t *dst, uint8_t* src, size_t size, bool gpu) {
        auto kind = gpu ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToHost;
        NdArray::memcpy(dst, src, size, kind);
    }

    static void memcpy(uint8_t *dst, uint8_t* src, size_t size, enum cudaMemcpyKind kind) {
        CUDA_ASSERT(cudaMemcpy(dst, src, size, kind));
    }

    static size_t getNumberOfElements(const DataShape &shape) {
        size_t result = 1;
        for (auto &val : shape) {
            result *= val;
        }
        return result;
    }

    void assertNotGpu(int line) const {
        if(gpu) {
            throw std::runtime_error(
                "NdArray, line " + std::to_string(line) + ":" +
                "Operation is not supported for GPU arrays.");
        }
    }

    uint8_t *ptr{nullptr};
    DataShape shape{};
    size_t nBytes{0};
    DataType dataType{DataType::UINT8};
    bool gpu{false};
    bool isExternal{false};
};
}// namespace arrus_example_imaging

#endif//CPP_EXAMPLE_NDARRAY_H
