#pragma once

#include "Controller/ControllerActions/ControllerAction.h"
#include "Controller/ControllerEvents/ControllerEvent.h"
#include "Controller/Controller.h"

class SetDeviceAction : public ControllerAction {
public:
    SetDeviceAction(Controller *controller);

    ~SetDeviceAction();

    void performAction(std::shared_ptr <IControllerEvent> controllerEvent);
};
