#ifndef ARRUS_CORE_DEVICES_US4R_BLOCKINGQUEUE_H
#define ARRUS_CORE_DEVICES_US4R_BLOCKINGQUEUE_H

#include <queue>
#include <condition_variable>

namespace arrus::devices {

template <typename T>
class BlockingQueue {
public:
    BlockingQueue(size_t maxSize) : maxSize(maxSize), isShutdown(false) {}

    bool enqueue(const T& item) {
        std::unique_lock<std::mutex> lock(stateMutex);
        if (isShutdown) {
            return false;
        }
        queueNotFull.wait(lock, [this]() { return queue.size() < maxSize || isShutdown; });
        if (isShutdown) {
            return false;
        }
        queue.push(item);
        queueNotEmpty.notify_one();
        return true;
    }

    std::optional<T> dequeue() {
        std::unique_lock<std::mutex> lock(stateMutex);
        queueNotEmpty.wait(lock, [this]() { return !queue.empty() || isShutdown; });
        if (isShutdown && queue.empty()) {
            return std::nullopt;
        }
        T item = queue.front();
        queue.pop();
        queueNotFull.notify_all();
        return item;
    }

    void shutdown() {
        std::unique_lock<std::mutex> lock(stateMutex);
        isShutdown = true;
        queueNotEmpty.notify_all();
        queueNotFull.notify_all();
    }

private:
    std::queue<T> queue;
    std::mutex stateMutex;
    std::condition_variable queueNotFull;
    std::condition_variable queueNotEmpty;
    size_t maxSize;
    bool isShutdown;
};

}

#endif