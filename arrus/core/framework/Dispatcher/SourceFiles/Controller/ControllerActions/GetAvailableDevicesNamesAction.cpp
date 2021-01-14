#include "Controller/ControllerActions/GetAvailableDevicesNamesAction.h"
#include "Controller/ControllerEvents/GetAvailableDevicesNamesEvent.h"
#include "Controller/Controller.h"

GetAvailableDevicesNamesAction::GetAvailableDevicesNamesAction(Controller* controller) : ControllerAction(controller)
{
	
}


GetAvailableDevicesNamesAction::~GetAvailableDevicesNamesAction()
{
}

void GetAvailableDevicesNamesAction::performAction(std::shared_ptr<IControllerEvent> controllerEvent)
{
	std::shared_ptr<GetAvailableDevicesNamesEvent> namesEvent = std::dynamic_pointer_cast<GetAvailableDevicesNamesEvent>(controllerEvent);
	namesEvent->setDevicesNames(this->controller->getModel()->getAvailableHALDevicesNames());
}
