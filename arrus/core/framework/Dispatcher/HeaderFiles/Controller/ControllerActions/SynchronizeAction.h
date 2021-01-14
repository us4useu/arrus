#pragma once

#include "Controller/ControllerActions/ControllerAction.h"
#include "Controller/Controller.h"

class SynchronizeAction : public ControllerAction {
public:
    SynchronizeAction(Controller *controller);

    ~SynchronizeAction();

    void performAction(std::shared_ptr <IControllerEvent> controllerEvent);
};

