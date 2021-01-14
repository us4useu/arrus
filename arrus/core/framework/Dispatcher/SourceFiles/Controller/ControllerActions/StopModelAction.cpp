#include "Controller/ControllerActions/StopModelAction.h"


StopModelAction::StopModelAction(Controller* controller) : ControllerAction(controller)
{
	
}


StopModelAction::~StopModelAction()
{
}

void StopModelAction::performAction(std::shared_ptr<IControllerEvent> controllerEvent)
{
	this->controller->getModel()->stop();
	this->controller->destroyModel();
}
