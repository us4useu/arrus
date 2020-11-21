#ifndef ARRUS_CORE_DEVICES_US4R_US4RHOSTBUFFER_H
#define ARRUS_CORE_DEVICES_US4R_US4RHOSTBUFFER_H

#include <mutex>

#include "arrus/core/api/devices/us4r/HostBuffer.h"

namespace arrus::devices {

class Us4RHostBuffer: public HostBuffer {
public:

    Us4RHostBuffer(size_t elementSize, uint16 nElements): nElements(nElements) {
        for(int i = 0; i < nElements; ++i) {
            auto elementPtr = (int16 *) operator new[](elementSize,std::align_val_t{4096});
            elements.push_back(elementPtr);
        }
    }

    ~Us4RHostBuffer() override {
        for(auto element: elements) {
            ::operator delete[](element, std::align_val_t{4096});
        }
    }

    /**
     * Returns true if the push was possible, false if the queue is closed.
     */
    bool push(const std::function<void(int16*)> &pushFunc) {
        {
            std::unique_lock<std::mutex> guard(mutex);
            if(this->isShutdown) {
                return false;
            }
            while(currentSize == elements.size()) {
                canPush.wait(guard);
                if(this->isShutdown) {
                    return false;
                }
            }
            pushFunc(elements[headIdx]);
            headIdx = (headIdx + 1) % nElements;
            ++currentSize;
        }
        canPop.notify_one();
        return true;
    }

    /**
     * @return a pointer when the access was possible, nullptr otherwise (e.g. queue shutdown).
     */
    int16 *tail(long long timeout) override {
        {
            std::unique_lock<std::mutex> guard(mutex);
            if(this->isShutdown) {
                return nullptr;
            }
            while(currentSize == 0) {
                ARRUS_WAIT_FOR_CV_OPTIONAL_TIMEOUT(
                    canPop, guard, timeout,
                    "Timeout while waiting for new data queue.")
                if(this->isShutdown) {
                    return nullptr;
                }
            }
            return elements[tailIdx];
        }
    }

    void releaseTail(long long timeout) override {
        {
            std::unique_lock<std::mutex> guard(mutex);
            while(currentSize == 0) {
                ARRUS_WAIT_FOR_CV_OPTIONAL_TIMEOUT(
                    canPop, guard, timeout,
                    "Timeout while waiting for new data queue.")
                if(this->isShutdown) {
                    throw IllegalStateException("The access to us4r buffer is closed.");
                }
            }
            tailIdx = (tailIdx + 1) % nElements;
            --currentSize;
        }
        canPush.notify_one();
    }

    void shutdown() {
        std::unique_lock<std::mutex> guard(mutex);
        this->isShutdown = true;
        this->canPush.notify_all();
        this->canPop.notify_all();
        guard.unlock();
    }

private:
    std::mutex mutex;
    std::condition_variable canPush;
    std::condition_variable canPop;
    uint16 nElements;
    std::vector<int16*> elements;
    // Tail - the latest added element, available to the consumer.
    uint16 tailIdx{0};
    // Head - the first free buffer element space available for new data.
    uint16 headIdx{0};
    size_t currentSize{0};
    bool isShutdown{false};
};
}

#endif //ARRUS_CORE_DEVICES_US4R_US4RHOSTBUFFER_H
