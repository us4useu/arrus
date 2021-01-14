#include "Controller/ControllerActions/SynchronizeAction.h"
#include "Controller/ControllerEvents/SynchronizeEvent.h"
#include "Model/GraphNodesLibrary/GraphNodes/CudaAssert.cuh"

SynchronizeAction::SynchronizeAction(Controller* controller) : ControllerAction(controller)
{
}


SynchronizeAction::~SynchronizeAction()
{
}

void SynchronizeAction::performAction(std::shared_ptr<IControllerEvent> controllerEvent)
{
	std::shared_ptr<SynchronizeEvent> syncEvent = std::dynamic_pointer_cast<SynchronizeEvent>(controllerEvent);
	this->controller->getModel()->synchronizeWithAllDevices();
	syncEvent->notify();
}
