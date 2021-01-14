#pragma once

#include "Controller/ControllerEvents/ControllerEvent.h"

class StopControllerEvent : public ControllerEvent<StopControllerEvent> {
public:
    StopControllerEvent() {};

    ~StopControllerEvent() {};
};

