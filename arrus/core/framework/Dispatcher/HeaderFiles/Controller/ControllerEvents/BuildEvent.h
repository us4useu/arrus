#pragma once

#include "Controller/ControllerEvents/ControllerEvent.h"

#include <string>

class BuildEvent : public ControllerEvent<BuildEvent> {
private:
    const std::string passingJson;
    const bool cascadeMode;
    bool succeeded;

public:
    BuildEvent(const std::string &json, const bool cascadeMode);

    ~BuildEvent();

    const std::string &getJson();

    const bool isCascadeMode();

    void setSucceeded(const bool val);

    bool getSucceeded();
};

