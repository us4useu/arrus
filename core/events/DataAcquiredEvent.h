#ifndef ARIUS_SDK_DATAACQUIREDEVENT_H
#define ARIUS_SDK_DATAACQUIREDEVENT_H

#include <cstddef>

#include "arius/core/events/Event.h"

namespace arius {
    class DataAcquiredEvent : public Event {
    public:
        DataAcquiredEvent(const size_t address, const size_t length) :
                address(address), length(length) {}

        ~DataAcquiredEvent() {}

        size_t getAddress() {
            return address;
        }

        size_t getLength() {
            return length;
        }

    private:
        size_t address;
        size_t length;
    };
}

#endif
