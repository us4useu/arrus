#pragma once

#include "Controller/ControllerActions/ControllerAction.h"
#include "Controller/Controller.h"

class StopControllerAction : public ControllerAction {
public:
    StopControllerAction(Controller *controller);

    ~StopControllerAction();

    void performAction(std::shared_ptr <IControllerEvent> controllerEvent);
};

