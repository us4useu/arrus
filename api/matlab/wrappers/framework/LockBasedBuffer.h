#ifndef API_MATLAB_WRAPPERS_FRAMEWORK_LOCKBASEDBUFFER_H
#define API_MATLAB_WRAPPERS_FRAMEWORK_LOCKBASEDBUFFER_H

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>

#include "arrus/core/api/arrus.h"

namespace arrus::matlab::framework {

/**
 * // front: oldest data
 * // back: latest data
 *
 * TODO This functionality should be implemented in ARRUS core.
 */
class LockBasedBuffer {
public:
    typedef std::optional<::arrus::framework::BufferElement::SharedHandle> BufferElementView;

    explicit LockBasedBuffer(::arrus::framework::DataBuffer *buffer) : buffer(buffer) {
        ::arrus::framework::OnNewDataCallback newDataCallback =
            [this](const ::arrus::framework::BufferElement::SharedHandle &element) { this->signal(element); };
        ::arrus::framework::OnShutdownCallback shutdownCallback = [this]() { this->shutdown(); };
        buffer->registerOnNewDataCallback(newDataCallback);
    }

    ~LockBasedBuffer() {
        shutdown();
    }

    BufferElementView front() {
        std::unique_lock<std::mutex> lock{this->mutex};
        while(queue.empty()) {
            this->dataReadyEvent.wait(lock);
            if(this->isShutdown) {
                return std::nullopt;
            }
        }
        auto value = queue.front();
        queue.pop_front();
        return value;
    }

    BufferElementView back() {
        std::unique_lock<std::mutex> lock{this->mutex};
        while(queue.empty()) {
            this->dataReadyEvent.wait(lock);
            if(this->isShutdown) {
                return std::nullopt;
            }
        }
        auto value = queue.back();
        queue.pop_back();
        return value;
    }

private:
    void signal(const ::arrus::framework::BufferElement::SharedHandle &element) {
        {
            std::unique_lock<std::mutex> lock{this->mutex};
            queue.push_back(element);
        }
        dataReadyEvent.notify_one();
    }

    void shutdown() {
        {
            std::unique_lock<std::mutex> lock{this->mutex};
            this->isShutdown = true;
        }
        dataReadyEvent.notify_all();
    }

    ::arrus::framework::DataBuffer *buffer;
    std::mutex mutex;
    std::condition_variable dataReadyEvent;
    std::deque<::arrus::framework::BufferElement::SharedHandle> queue;
    // True, when buffer was already closed.
    bool isShutdown{false};
};

}// namespace arrus::matlab

#endif//API_MATLAB_WRAPPERS_FRAMEWORK_LOCKBASEDBUFFER_H
