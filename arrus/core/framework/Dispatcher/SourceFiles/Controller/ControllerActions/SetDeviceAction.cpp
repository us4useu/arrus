#include "Controller/ControllerActions/SetDeviceAction.h"
#include "Controller/ControllerEvents/SetDeviceEvent.h"
#include "Controller/Controller.h"

SetDeviceAction::SetDeviceAction(Controller* controller) : ControllerAction(controller)
{
	
}


SetDeviceAction::~SetDeviceAction()
{
}

void SetDeviceAction::performAction(std::shared_ptr<IControllerEvent> controllerEvent)
{
	std::shared_ptr<SetDeviceEvent> setDeviceEvent = std::dynamic_pointer_cast<SetDeviceEvent>(controllerEvent);
	this->controller->getModel()->setChosenHALDeviceName(setDeviceEvent->getDeviceName());
}
