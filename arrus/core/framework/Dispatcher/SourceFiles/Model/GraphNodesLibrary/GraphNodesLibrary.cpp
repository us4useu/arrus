#include "Model/GraphNodesLibrary/GraphNodesLibrary.h"
#include "Utils/DispatcherLogger.h"

GraphNodesLibrary::GraphNodesLibrary()
{
}


GraphNodesLibrary::~GraphNodesLibrary()
{
}

void GraphNodesLibrary::registerNode(const nodeId id, const std::shared_ptr<GraphNode> node)
{
	if (this->graphNodes.find(id) != this->graphNodes.end())
		DISPATCHER_LOG(DispatcherLogType::WARNING, std::string("Node with id: ") + std::to_string(id) + std::string(" already exists."));
	this->graphNodes.insert(std::make_pair(id, node));
}

std::shared_ptr<GraphNode> GraphNodesLibrary::getNode(const nodeId id)
{
	graphNodesMapType::iterator it = this->graphNodes.find(id);
	if (it == this->graphNodes.end())
	{
		DISPATCHER_LOG(DispatcherLogType::ERROR_, std::string("Node with id: ") + std::to_string(id) + std::string(" doesn't exist."));
		return nullptr;
	}
	return it->second;
}

void GraphNodesLibrary::connectNodes(const nodeId srcId, const nodeId dstId)
{
	std::shared_ptr<GraphNode> srcNode = this->getNode(srcId);
	std::shared_ptr<GraphNode> dstNode = this->getNode(dstId);
	if ((srcNode != nullptr) && (dstNode != nullptr))
	{
		srcNode->addSuccessor(dstNode);
		dstNode->addPredecessor(srcNode);
	}
	else
		DISPATCHER_LOG(DispatcherLogType::ERROR_, std::string("Cannot connect node: ") + std::to_string(srcId) + std::string(" with node: ") + std::to_string(dstId));
}

void GraphNodesLibrary::updateGraphNodeParameter(const nodeId id, const parameterName& name, const parameterValue& value)
{
	boost::mutex::scoped_lock lock(this->mutex);

	this->pendingUpdates.insert(std::make_pair(std::make_pair(id, name), value));
}

void GraphNodesLibrary::applyGraphNodesUpdates()
{
	boost::mutex::scoped_lock lock(this->mutex);

	for (pendingGraphNodesUpdatesMap::iterator it = this->pendingUpdates.begin(); it != this->pendingUpdates.end(); ++it)
	{
		std::shared_ptr<GraphNode> node = this->getNode(it->first.first);
		node->setNodeVariable(it->first.second, it->second);
	}

	this->pendingUpdates.clear();
}