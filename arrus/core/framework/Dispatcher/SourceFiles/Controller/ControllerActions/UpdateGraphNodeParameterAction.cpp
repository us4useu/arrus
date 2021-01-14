#include "Controller/ControllerActions/UpdateGraphNodeParameterAction.h"
#include "Controller/ControllerEvents/UpdateGraphNodeParameterEvent.h"

UpdateGraphNodeParameterAction::UpdateGraphNodeParameterAction(Controller* controller) : ControllerAction(controller)
{
	
}

UpdateGraphNodeParameterAction::~UpdateGraphNodeParameterAction()
{

}

void UpdateGraphNodeParameterAction::performAction(std::shared_ptr<IControllerEvent> controllerEvent)
{
	std::shared_ptr<UpdateGraphNodeParameterEvent> updateGraphNodeParameterEvent = std::dynamic_pointer_cast<UpdateGraphNodeParameterEvent>(controllerEvent);

	int graphsNumber = this->controller->getModel()->getGraphNodesLibraryNumber();
	for (int i = 0; i < graphsNumber; ++i)
		this->controller->getModel()->getGraphNodesLibrary(i)->updateGraphNodeParameter(updateGraphNodeParameterEvent->getNodeId(), updateGraphNodeParameterEvent->getParameterName(),
																	   updateGraphNodeParameterEvent->getParameterValue());
}