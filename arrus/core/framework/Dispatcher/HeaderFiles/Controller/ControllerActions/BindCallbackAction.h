#pragma once

#include "Controller/ControllerActions/ControllerAction.h"
#include "Controller/ControllerEvents/ControllerEvent.h"
#include "Controller/Controller.h"

class BindCallbackAction : public ControllerAction {
public:
    BindCallbackAction(Controller *controller);

    ~BindCallbackAction();

    void performAction(std::shared_ptr <IControllerEvent> controllerEvent);
};
