#include "Model/GraphNodesLibrary/GraphNodesFactory.h"
#include "Utils/DispatcherLogger.h"

std::shared_ptr<nodesMapType> GraphNodesFactory::nodesConstructors = std::shared_ptr<nodesMapType>(new nodesMapType());

std::shared_ptr<GraphNode> GraphNodesFactory::createGraphNode(const nodeNameType &nodeName)
{
	nodesMapType::iterator it = nodesConstructors->find(nodeName);
	if (it == nodesConstructors->end())
	{
		DISPATCHER_LOG(DispatcherLogType::ERROR_, std::string("Node with name: ") + nodeName + std::string(" cannot be created because it doesn't exist."));
		return nullptr;
	}
	return it->second();
}