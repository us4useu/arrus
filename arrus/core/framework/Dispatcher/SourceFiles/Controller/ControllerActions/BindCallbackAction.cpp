#include "Controller/ControllerActions/BindCallbackAction.h"
#include "Controller/ControllerEvents/BindCallbackEvent.h"
#include <boost/function.hpp>
#include "Utils/DispatcherLogger.h"

BindCallbackAction::BindCallbackAction(Controller* controller) : ControllerAction(controller)
{
	
}


BindCallbackAction::~BindCallbackAction()
{
}

void BindCallbackAction::performAction(std::shared_ptr<IControllerEvent> controllerEvent)
{
	std::shared_ptr<BindCallbackEvent> bindEvent = std::dynamic_pointer_cast<BindCallbackEvent>(controllerEvent);
	int graphsNumber = this->controller->getModel()->getGraphNodesLibraryNumber();
	for (int i = 0; i < graphsNumber; ++i)
	{
		std::shared_ptr<GraphNode> node = this->controller->getModel()->getGraphNodesLibrary(i)->getNode(bindEvent->getNodeId());
		if (node != nullptr)
			node->registerCallback(bindEvent->getCallback());
		else
		{
			DISPATCHER_LOG(DispatcherLogType::WARNING, std::string("Due to previous errors callback couldn't be installed."));
			return;
		}
	}
}
 