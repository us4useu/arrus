#ifndef ARRUS_CORE_DEVICES_US4R_US4R_EVENT_H
#define ARRUS_CORE_DEVICES_US4R_US4R_EVENT_H

#include <string>

namespace arrus::devices {

class Us4REvent {
public:
    Us4REvent(std::string id) : id(id) {}

    const std::string &getId() const { return id; }

private:
    std::string id;
};
}

#endif