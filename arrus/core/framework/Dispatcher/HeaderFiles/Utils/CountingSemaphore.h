#pragma once

#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/lock_types.hpp>

class CountingSemaphore {
    unsigned int count;
    boost::mutex mutex;
    boost::condition_variable condition;

public:
    explicit CountingSemaphore(unsigned int initial_count = 0) : count(initial_count), mutex(), condition() {}

    unsigned int get_count() {
        boost::unique_lock <boost::mutex> lock(mutex);
        return count;
    }

    void signal() {
        boost::unique_lock <boost::mutex> lock(mutex);

        ++count;

        condition.notify_one();
    }

    void wait() {
        boost::unique_lock <boost::mutex> lock(mutex);
        while(count == 0) {
            condition.wait(lock);
        }
        --count;
    }
};