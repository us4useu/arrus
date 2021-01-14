#pragma once

#include "Controller/ControllerEvents/ControllerEvent.h"
#include <vector>
#include <string>

class GetAvailableDevicesNamesEvent : public ControllerEvent<GetAvailableDevicesNamesEvent> {
private:
    std::vector <std::string> devicesNames;

public:
    GetAvailableDevicesNamesEvent();

    ~GetAvailableDevicesNamesEvent();

    std::vector <std::string> getDevicesNames();

    void setDevicesNames(const std::vector <std::string> &devicesNames);
};

