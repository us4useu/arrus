#ifndef ARRUS_ARRUS_CORE_SESSION_BLOCKINGQUEUE_H
#define ARRUS_ARRUS_CORE_SESSION_BLOCKINGQUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>

namespace arrus {

template<typename  T>
class BlockingQueue {
public:

    BlockingQueue() = default;
    ~BlockingQueue() = default;

    void push(const T& value) {
        {
            std::unique_lock<std::mutex> lock(mutex);
            this->queue.push_front(value);
        }
        this->dataAvailable.notify_one();
    }

    bool pop(T &output) {
        std::unique_lock<std::mutex> lock(mutex);
        while(queue.empty()) {
            if(this->isShutdown) {
                return false;
            }
            dataAvailable.wait(lock);
        }
        output = std::move(this->queue.front());
        this->queue.pop();
        return true;
    }

    void shutdown() {
        {
            std::unique_lock<std::mutex> lock(mutex);
            this->isShutdown = true;
        }
        this->dataAvailable.notify_all();
    }

private:
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable dataAvailable;
    bool isShutdown{false};
};

}

#endif //ARRUS_ARRUS_CORE_SESSION_BLOCKINGQUEUE_H
