#pragma once

#include "Controller/ControllerActions/ControllerAction.h"
#include "Controller/Controller.h"

class UpdateGraphNodeParameterAction : public ControllerAction {
public:
    UpdateGraphNodeParameterAction(Controller *controller);

    ~UpdateGraphNodeParameterAction();

    void performAction(std::shared_ptr <IControllerEvent> controllerEvent);
};