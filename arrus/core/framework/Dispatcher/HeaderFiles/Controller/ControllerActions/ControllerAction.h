#pragma once

#include "Controller/ControllerEvents/ControllerEvent.h"
#include <memory>

class Controller;

class ControllerAction {
protected:
    Controller *controller;

public:
    ControllerAction(Controller *controller) : controller(controller) {};

    virtual ~ControllerAction() {};

    virtual void performAction(std::shared_ptr <IControllerEvent> controllerEvent) {};
};