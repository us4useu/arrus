#include "Controller/ControllerActions/StartModelAction.h"
#include "Controller/ControllerEvents/StartModelEvent.h"
#include "Controller/Controller.h"

StartModelAction::StartModelAction(Controller* controller) : ControllerAction(controller)
{
	
}


StartModelAction::~StartModelAction()
{
}

void StartModelAction::performAction(std::shared_ptr<IControllerEvent> controllerEvent)
{
	std::shared_ptr<StartModelEvent> startEvent = std::dynamic_pointer_cast<StartModelEvent>(controllerEvent);
	this->controller->getModel()->start(startEvent->doesStartOnce());
}
