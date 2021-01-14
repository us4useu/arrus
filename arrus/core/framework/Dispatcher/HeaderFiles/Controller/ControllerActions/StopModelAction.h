#pragma once

#include "Controller/ControllerActions/ControllerAction.h"
#include "Controller/Controller.h"

class StopModelAction : public ControllerAction {
public:
    StopModelAction(Controller *controller);

    ~StopModelAction();

    void performAction(std::shared_ptr <IControllerEvent> controllerEvent);
};

