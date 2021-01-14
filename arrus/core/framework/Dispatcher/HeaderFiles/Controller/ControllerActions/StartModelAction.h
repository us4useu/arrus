#pragma once

#include "Controller/ControllerActions/ControllerAction.h"
#include "Controller/ControllerEvents/ControllerEvent.h"
#include "Controller/Controller.h"

class StartModelAction : public ControllerAction {
public:
    StartModelAction(Controller *controller);

    ~StartModelAction();

    void performAction(std::shared_ptr <IControllerEvent> controllerEvent);
};

