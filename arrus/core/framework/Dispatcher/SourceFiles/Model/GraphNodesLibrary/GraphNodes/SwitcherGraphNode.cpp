#include "Model/GraphNodesLibrary/GraphNodes/SwitcherGraphNode.h"
#include "Utils/DispatcherLogger.h"
#include <algorithm>

GraphNodesFactoryRegister<SwitcherGraphNode> SwitcherGraphNode::graphNodesFactoryRegister("switcher");

SwitcherGraphNode::SwitcherGraphNode()
{
	this->setNodeVariable("matcher", VariableAnyValue(std::unordered_map<std::string, VariableAnyValue>()));
}

SwitcherGraphNode::~SwitcherGraphNode()
{
}

std::vector<std::shared_ptr<GraphNode>>::iterator SwitcherGraphNode::findSuccessorWithNodeId(const int successorNodeId)
{
	for (std::vector<std::shared_ptr<GraphNode>>::iterator it = this->successors.begin(); it != this->successors.end(); ++it)
	{
		if ((*it)->getNodeId() == successorNodeId)
			return it;
	}
	return this->successors.end();
}

void SwitcherGraphNode::pushSuccessorToFront(const int successorNodeId)
{
	std::vector<std::shared_ptr<GraphNode>>::iterator successorIt = this->findSuccessorWithNodeId(successorNodeId);
	if (successorIt == this->successors.end())
	{
		DISPATCHER_LOG(DispatcherLogType::ERROR_, std::string("Switcher node with id: ") + std::to_string(this->getNodeId()) 
			+ std::string(" is not connected with desired node with id: ") + std::to_string(successorNodeId));
		DISPATCHER_LOG(DispatcherLogType::WARNING, std::string("Due to previous error next node in graph flow has id: " + std::to_string(this->successors.front()->getNodeId())));
	}
	else
	{
		std::iter_swap(successors.begin(), successorIt);
	}
}

void SwitcherGraphNode::process(cudaStream_t& defaultStream)
{
	std::unordered_map<std::string, VariableAnyValue> matcher = this->getNodeVariable("matcher").getValue<std::unordered_map<std::string, VariableAnyValue>>();
	int frameId = this->inputData.getPtrProperty("frameId").getValue<int>();

	std::string matcherKey = std::to_string(frameId);
	std::unordered_map<std::string, VariableAnyValue>::iterator it = matcher.find(matcherKey);
	if (it == matcher.end())
	{
		DISPATCHER_LOG(DispatcherLogType::ERROR_, std::string("Switcher node with id: ") + std::to_string(this->getNodeId()) + std::string(" doesn't have key with value: ") + matcherKey);
		DISPATCHER_LOG(DispatcherLogType::WARNING, std::string("Due to previous error next node in graph flow has id: " + std::to_string(this->successors.front()->getNodeId())));
	}
	else
	{
		int dstNodeId = it->second.getValue<int>();
		this->pushSuccessorToFront(dstNodeId);
	}

	this->outputData = this->inputData;
}