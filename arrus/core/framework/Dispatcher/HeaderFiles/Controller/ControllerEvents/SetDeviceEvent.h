#pragma once

#include "Controller/ControllerEvents/ControllerEvent.h"
#include <vector>
#include <string>

class SetDeviceEvent : public ControllerEvent<SetDeviceEvent> {
private:
    std::string deviceName;

public:
    SetDeviceEvent(const std::string &deviceName);

    ~SetDeviceEvent();

    std::string &getDeviceName();
};

