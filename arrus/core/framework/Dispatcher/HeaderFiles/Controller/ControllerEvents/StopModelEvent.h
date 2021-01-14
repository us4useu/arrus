#pragma once

#include "Controller/ControllerEvents/ControllerEvent.h"

class StopModelEvent : public ControllerEvent<StopModelEvent> {
public:
    StopModelEvent() {};

    ~StopModelEvent() {};
};

