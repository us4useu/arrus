#pragma once

#include "Controller/ControllerEvents/ControllerEvent.h"
#include <boost/thread/mutex.hpp>

class SynchronizeEvent : public ControllerEvent<SynchronizeEvent> {
private:
    mutable boost::mutex mutex;
public:
    SynchronizeEvent();

    ~SynchronizeEvent();

    void wait();

    void notify();
};

