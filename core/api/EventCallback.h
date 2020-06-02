#ifndef ARIUS_SDK_EVENTCALLBACK_H
#define ARIUS_SDK_EVENTCALLBACK_H

namespace arrus {
    class EventCallback {
        virtual void run(const Event& e) = 0;
    };
}

#endif
