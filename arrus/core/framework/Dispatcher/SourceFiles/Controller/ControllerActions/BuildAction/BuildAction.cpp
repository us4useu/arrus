#include "Controller/ControllerActions/BuildAction/BuildAction.h"
#include "Controller/ControllerActions/BuildAction/BuildConfigurationCheck.h"
#include "Controller/ControllerEvents/BuildEvent.h"
#include "Model/GraphNodesLibrary/GraphNodesFactory.h"
#include <boost/foreach.hpp>
#include <regex>
#include <sstream>
#include "Utils/DispatcherLogger.h"

BuildAction::BuildAction(Controller* controller) : ControllerAction(controller)
{
	
}

BuildAction::~BuildAction()
{
}

void BuildAction::performAction(std::shared_ptr<IControllerEvent> controllerEvent)
{
	std::shared_ptr<BuildEvent> buildEvent = std::dynamic_pointer_cast<BuildEvent>(controllerEvent);
	if (buildEvent->isCascadeMode())
		this->controller->getModel()->setCascadeMode();
	if (!this->parseJson(buildEvent->getJson()))
	{
		buildEvent->setSucceeded(false);
		return;
	}
	this->controller->getModel()->setJsonConfig(buildEvent->getJson());
}

VariableAnyValue BuildAction::parseSingleVariableValue(const boost::property_tree::ptree &pt, const int graphIndex)
{
	std::smatch pointer;
	const std::string value = pt.data();

	// Note that function uses regex instead of casting in order to avoid exceptions

	// #<graphNodeId>#<variableName> example: "#12#height" 
	static const std::regex pointer_regex = std::regex("^#([0-9]+)#(.+)$");
	// Just integer 
	static const std::regex int_regex = std::regex("^[+-]?[0-9]+$");
	// Checking for float (note that C++ needs escaped backslash (\\)
	static const std::regex float_regex = std::regex("^[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?$");

	static const std::regex bool_regex = std::regex("^(true)|(false)$");

	if (std::regex_search(value, pointer, pointer_regex))
	{
		const int graphNodeId = std::stoi(pointer[1]);
		const std::string variableName = pointer[2];

		std::shared_ptr<GraphNode> graphNode = this->controller->getModel()->getGraphNodesLibrary(graphIndex)->getNode(graphNodeId);
		if (graphNode == nullptr)
		{
			DISPATCHER_LOG(DispatcherLogType::ERROR_, std::string("Cannot create pointer to variable: ") + variableName + std::string(" from node with id: ") + std::to_string(graphNodeId));
			return VariableAnyValue();
		}
		return VariableAnyValue(graphNode->getNodeVariable(variableName).getAnyValuePtr(), true);
	}
	else if (std::regex_match(value, int_regex)) {
		return VariableAnyValue(pt.get_value<int>());
	}
	else if (std::regex_match(value, float_regex)) {
		return VariableAnyValue(pt.get_value<float>());
	}
	else if (std::regex_match(value, bool_regex)) {
		return VariableAnyValue(pt.get_value<bool>());
	}
	// Assume plain string
	else {
		return VariableAnyValue(value);
	}
}

VariableAnyValue BuildAction::parseCompoundVariableValue(const boost::property_tree::ptree &pt, const int graphIndex)
{
	if (std::string(pt.front().first.data()).empty())
	{
		std::vector<VariableAnyValue> nodeVariables;
		BOOST_FOREACH(const boost::property_tree::ptree::value_type &vKey, pt)
		{
			VariableAnyValue varVal = this->getNodeVariableValue(vKey.second, graphIndex);
			nodeVariables.push_back(varVal);
		}
		return VariableAnyValue(nodeVariables);
	}

	return VariableAnyValue(this->getNodeVariables(pt, graphIndex));
}

VariableAnyValue BuildAction::getNodeVariableValue(const boost::property_tree::ptree &pt, const int graphIndex)
{	
	if (pt.data().empty())
		return this->parseCompoundVariableValue(pt, graphIndex);
	
	return this->parseSingleVariableValue(pt, graphIndex);
}

std::unordered_map<variableName, VariableAnyValue> BuildAction::getNodeVariables(const boost::property_tree::ptree &pt, const int graphIndex)
{
	std::unordered_map<variableName, VariableAnyValue> nodeVariables;

	BOOST_FOREACH(const boost::property_tree::ptree::value_type &vKey, pt)
	{
		variableName varName = vKey.first.data();
		VariableAnyValue varVal = this->getNodeVariableValue(vKey.second, graphIndex);
		nodeVariables.insert(std::make_pair(varName, varVal));
	}

	return nodeVariables;
}

void BuildAction::recursiveMergeOfNodesVariables(std::unordered_map<variableName, VariableAnyValue>* oldVals,
	const std::unordered_map<variableName, VariableAnyValue>& newVals)
{
	for (const std::pair<variableName, VariableAnyValue>& keyVal : newVals)
	{
		PropertyType varType = keyVal.second;
		if (varType == PropertyType::InnerType::MAP)
			this->recursiveMergeOfNodesVariables((*oldVals)[keyVal.first].getValuePtr<std::unordered_map<variableName, VariableAnyValue>>(), 
				keyVal.second.getValue<const std::unordered_map<variableName, VariableAnyValue>>());
		else
			(*oldVals)[keyVal.first] = keyVal.second;
	}
}

void BuildAction::parseGraphNodes(const boost::property_tree::ptree &pt, const int graphIndex)
{
	const boost::property_tree::ptree graphNodes = pt.get_child("graphNodes");

	BOOST_FOREACH(const boost::property_tree::ptree::value_type &v, graphNodes)
	{
		std::string nodeName = v.second.get<std::string>("name");
		// TODO instead of using json here, create graph nodes according to passed Pipeline object
		std::shared_ptr<GraphNode> graphNode = GraphNodesFactory::createGraphNode(nodeName);
		int nodeId = v.second.get<int>("id");
		graphNode->setNodeId(nodeId);
		this->controller->getModel()->getGraphNodesLibrary(graphIndex)->registerNode(nodeId, graphNode);
		std::unordered_map<variableName, VariableAnyValue> defaultValues = graphNode->getNodeVariables();
		this->recursiveMergeOfNodesVariables(&defaultValues, this->getNodeVariables(v.second.get_child("variables"), graphIndex));
		graphNode->setNodeVariables(defaultValues);
	}
}

void BuildAction::parseGraphNodesConnections(const boost::property_tree::ptree &pt, const int graphIndex)
{
	const boost::property_tree::ptree connections = pt.get_child("graphNodesConnections");
	
	BOOST_FOREACH(const boost::property_tree::ptree::value_type &v, connections)
	{
		int srcId = v.second.get<int>("srcId");
		int dstId = v.second.get<int>("dstId");
		this->controller->getModel()->getGraphNodesLibrary(graphIndex)->connectNodes(srcId, dstId);
	}
}

bool BuildAction::parseJson(const std::string &json)
{
	BuildConfigurationCheck bcc(json);
	if (!bcc.performCheck())
		return false;

	boost::property_tree::ptree pt = bcc.getParsedJson();

	int numberOfGraphs = this->controller->getModel()->getGraphNodesLibraryNumber();
	for (int graphIndex = 0; graphIndex < numberOfGraphs; ++graphIndex)
	{
		this->parseGraphNodes(pt, graphIndex);
		this->parseGraphNodesConnections(pt, graphIndex);
	}
	return true;
}