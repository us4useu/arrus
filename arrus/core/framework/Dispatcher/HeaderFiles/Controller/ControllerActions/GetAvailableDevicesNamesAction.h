#pragma once

#include "Controller/ControllerActions/ControllerAction.h"
#include "Controller/ControllerEvents/ControllerEvent.h"
#include "Controller/Controller.h"

class GetAvailableDevicesNamesAction : public ControllerAction {
public:
    GetAvailableDevicesNamesAction(Controller *controller);

    ~GetAvailableDevicesNamesAction();

    void performAction(std::shared_ptr <IControllerEvent> controllerEvent);
};
