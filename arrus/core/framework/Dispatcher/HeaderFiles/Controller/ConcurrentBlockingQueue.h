#pragma once

#include <queue>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>

template<typename Data>
class ConcurrentBlockingQueue {
private:
    std::queue <Data> queue;
    mutable boost::mutex mutex;
    boost::condition_variable condition;
public:
    ConcurrentBlockingQueue() {};

    ~ConcurrentBlockingQueue() {};

    void push(const Data &data) {
        boost::mutex::scoped_lock lock(this->mutex);
        this->queue.push(data);
        lock.unlock();

        this->condition.notify_one();
    }

    Data pop() {
        boost::mutex::scoped_lock lock(this->mutex);
        while(this->queue.empty()) // while - to guard against spurious wakeups
        {
            this->condition.wait(lock);
        }

        Data data = this->queue.front();
        this->queue.pop();
        return data;
    }
};

