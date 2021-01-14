#pragma once

#include "Controller/ControllerEvents/ControllerEvent.h"

class StartModelEvent : public ControllerEvent<StartModelEvent> {
private:
    bool startOnce;
public:
    StartModelEvent(const bool startOnce = false);

    ~StartModelEvent();

    const bool doesStartOnce();
};

