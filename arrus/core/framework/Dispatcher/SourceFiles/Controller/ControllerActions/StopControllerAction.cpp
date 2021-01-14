#include "Controller/ControllerActions/StopControllerAction.h"


StopControllerAction::StopControllerAction(Controller* controller) : ControllerAction(controller)
{
	
}


StopControllerAction::~StopControllerAction()
{
}

void StopControllerAction::performAction(std::shared_ptr<IControllerEvent> controllerEvent)
{
	this->controller->stopWork();
}
