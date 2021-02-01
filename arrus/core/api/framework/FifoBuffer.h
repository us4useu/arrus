#ifndef ARRUS_ARRUS_CORE_API_FRAMEWORK_FIFOBUFFER_H
#define ARRUS_ARRUS_CORE_API_FRAMEWORK_FIFOBUFFER_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <utility>

#include "arrus/core/api/common/exceptions.h"
#include "arrus/core/api/framework/FifoLockFreeBuffer.h"

namespace arrus::framework {

/**
 * Wraps FifoLockFree buffer into a blocking queue.
 *
 * This function provides functions that makes it possible to handle new data in the caller thread.
 */
class FifoBuffer {
public:
    explicit FifoBuffer(FifoLockFreeBuffer::SharedHandle inputBuffer) : inputBuffer(std::move(inputBuffer)) {
        OnNewDataCallback callback = [this] (const DataBufferElement::SharedHandle& dataPtr) {
            this->push(dataPtr);
        };
        OnOverflowCallback overflowCallback = [this] () {
            this->markAsInvalid();
        };
        OnShutdownCallback shutdownCallback = [this] () {
            this->shutdown();
        };
        this->inputBuffer->registerOnNewDataCallback(callback);
        this->inputBuffer->registerOnOverflowCallback(overflowCallback);
    }

    void push(const DataBufferElement::SharedHandle &element) {
        {
            std::unique_lock<std::mutex> lock(mutex);
            queue.push(element);
        }
        onNewData.notify_one();
    }

    /**
     * Returns true, if the queue was shut down, false otherwise.
     */
    std::pair<bool, DataBufferElement::SharedHandle> pop() {
        std::unique_lock<std::mutex> lock(mutex);
        while(true) {
            if(isInvalid) {
                throw IllegalStateException("Buffer is in invalid state, "
                                            "probably some data overflow happened.");
            }
            if(queue.empty()) {
                if(isShutdown) {
                    return {true, nullptr};
                }
            }
            else {
                break;
            }
            onNewData.wait(lock);
        }
        auto dataPtr = queue.front();
        queue.pop();
        return {false, dataPtr};
    }

    void shutdown() {
        {
            std::unique_lock<std::mutex> lock(mutex);
            isShutdown = true;
        }
        onNewData.notify_all();
    }

    void markAsInvalid() {
        {
            std::unique_lock<std::mutex> lock(mutex);
            isInvalid = true;
        }
        onNewData.notify_all();
    }

private:
    FifoLockFreeBuffer::SharedHandle inputBuffer;
    std::queue<DataBufferElement::SharedHandle> queue;
    std::condition_variable onNewData;
    std::mutex mutex;
    bool isShutdown{false};
    bool isInvalid{false};
};

}

#endif //ARRUS_ARRUS_CORE_API_FRAMEWORK_FIFOBUFFER_H