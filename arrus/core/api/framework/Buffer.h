#ifndef ARRUS_CORE_API_FRAMEWORK_H
#define ARRUS_CORE_API_FRAMEWORK_H

namespace arrus::framework {

class BufferElement {
public:
    using SharedHandle = std::shared_ptr<BufferElement>;

    virtual ~BufferElement() = default;

    virtual void release() = 0;

    virtual NDArray& getData() = 0;
};

class Buffer {
public:
    using Handle = std::unique_ptr<Buffer>;
    using SharedHandle = std::shared_ptr<Buffer>;

    virtual ~HostBuffer() = default;

    virtual unsigned short getNumberOfElements() const = 0;

    virtual size_t getElementSize() const = 0;
};

}

#endif //ARRUS_CORE_API_FRAMEWORK_H
