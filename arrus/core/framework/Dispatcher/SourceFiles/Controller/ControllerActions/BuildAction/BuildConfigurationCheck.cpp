#include "Controller/ControllerActions/BuildAction/BuildConfigurationCheck.h"
#include "Utils/DispatcherLogger.h"
#include <boost/property_tree/json_parser.hpp>
#include "jsonConfigParser.h"

BuildConfigurationCheck::BuildConfigurationCheck(const std::string& jsonConfiguration)
{
	this->jsonConfiguration = jsonConfiguration;
}

const boost::property_tree::ptree& BuildConfigurationCheck::getParsedJson()
{
	return this->parsedJsonConfiguration;
}

void BuildConfigurationCheck::parseJsonConfiguration()
{
	std::istringstream jsonStream(this->jsonConfiguration);
	try
	{
		boost::property_tree::read_json(jsonStream, this->parsedJsonConfiguration);
	}
	catch (boost::property_tree::json_parser_error exception)
	{
		throw BuildConfigurationCheck::CheckError("Error occured during parsing json config file. Line: " + 
			std::to_string(exception.line()) + std::string(" message: ") + exception.message());
	}
}

void BuildConfigurationCheck::validateHALJsonConfiguration()
{
	JsonConfigParser jsp;
	if (jsp.validateJson(this->jsonConfiguration) != IHAL::HAL_ERRORS::HAL_OK)
		throw BuildConfigurationCheck::CheckError(jsp.getLastJsonValidationError());
}

void BuildConfigurationCheck::checkChildren(const boost::property_tree::ptree& parent, std::unordered_map<std::string, PropertyType> properties)
{	
	for (const boost::property_tree::ptree::value_type& child : parent)
	{
		std::unordered_map<std::string, PropertyType>::iterator it = properties.find(child.first);
		
		if (it == properties.end())
			throw BuildConfigurationCheck::CheckError(std::string("Build configuration: Unknown property name has been found: ") + child.first);
		else
		{
			PropertyType detectedType(child.second);
			if (detectedType != it->second)
				throw BuildConfigurationCheck::CheckError(std::string("Build configuration: Wrong property data type in: ") + child.first +
					std::string(". Found ") + detectedType.toString() + std::string(" instead of ") + it->second.toString());

			if ((detectedType == PropertyType::InnerType::MAP_OR_ARRAY_OR_STRING) && (it->first.compare("variables") != 0))
				throw BuildConfigurationCheck::CheckError(std::string("Build configuration: Missing value for variable named: ") + it->first);
		}
	}
}

bool BuildConfigurationCheck::performCheck()
{
	try
	{
		this->validateHALJsonConfiguration();
		this->parseJsonConfiguration();

		std::unordered_map<std::string, PropertyType> properties;
		properties["transducer"] = PropertyType::InnerType::MAP;
		properties["hal"] = PropertyType::InnerType::MAP;
		properties["graphNodes"] = PropertyType::InnerType::ARRAY;
		properties["graphNodesConnections"] = PropertyType::InnerType::ARRAY;
		this->checkChildren(this->parsedJsonConfiguration, properties);

		boost::optional<boost::property_tree::ptree&> graphNodes = this->parsedJsonConfiguration.get_child_optional("graphNodes");
		std::vector<int> nodesIds;
		if (graphNodes)
			nodesIds = this->checkGraphNodes(graphNodes.get());
		else
			throw BuildConfigurationCheck::CheckError(std::string("Build configuration: GraphNodes property has not been found."));

		boost::optional<boost::property_tree::ptree&> graphNodesConnections = this->parsedJsonConfiguration.get_child_optional("graphNodesConnections");
		if (graphNodesConnections)
			this->checkGraphNodesConnections(graphNodesConnections.get(), nodesIds, graphNodes.get());
		else
			throw BuildConfigurationCheck::CheckError(std::string("Build configuration: GraphNodesConnections property has not been found."));
	}
	catch (const BuildConfigurationCheck::CheckError& error)
	{
		DISPATCHER_LOG(DispatcherLogType::FATAL, error.errorMessage);
		return false;
	}

	return true;
}

std::vector<int> BuildConfigurationCheck::checkGraphNodesIds(const boost::property_tree::ptree& parent)
{
	std::vector<int> ids;
	for (const boost::property_tree::ptree::value_type& child : parent)
	{
		int id = child.second.get_child("id").get_value<int>();
		
		if (id == 0)
			throw BuildConfigurationCheck::CheckError(std::string("Build configuration: Id with 0 value is reserved by the Framework."));

		std::vector<int>::iterator it;
		for (it = ids.begin(); it != ids.end(); ++it)
		{
			if (*it == id)
				throw BuildConfigurationCheck::CheckError(std::string("Build configuration: Duplicated node id with value: ") + std::to_string(id));

			if (*it > id)
				break;
		}
		ids.insert(it, id);
	}
	return ids;
}

void BuildConfigurationCheck::checkGraphNodeVariables(const boost::property_tree::ptree& pt, std::unordered_map<variableName, VariableAnyValue>& nodeVariables)
{
	std::unordered_map<std::string, PropertyType> nodesVariablesProperties;
	for (std::pair<variableName, VariableAnyValue> keyVal : nodeVariables)
		nodesVariablesProperties[keyVal.first] = keyVal.second;

	this->checkChildren(pt, nodesVariablesProperties);

	for (const std::pair<std::string, PropertyType>& keyVal : nodesVariablesProperties)
	{
		if (keyVal.second == PropertyType::InnerType::MAP)
		{
			std::unordered_map<variableName, VariableAnyValue> varMap = nodeVariables[keyVal.first].getValue<std::unordered_map<variableName, VariableAnyValue>>();
			// if default values of nested variables map are not specified then they are not checked (see switcher node)
			if (!varMap.empty() && (pt.count(keyVal.first) != 0))
				this->checkGraphNodeVariables(pt.get_child(keyVal.first), varMap);
		}
	}
}

void BuildConfigurationCheck::checkGraphNodesNamesAndVariables(const boost::property_tree::ptree& parent)
{
	for (const boost::property_tree::ptree::value_type& child : parent)
	{
		std::string nodeName = child.second.get_child("name").get_value<std::string>();
		std::shared_ptr<GraphNode> graphNode = GraphNodesFactory::createGraphNode(nodeName);
		if (graphNode == nullptr)
			throw BuildConfigurationCheck::CheckError(std::string("Build configuration: Unknown node name: ") + nodeName);

		if (nodeName.compare("plugin") == 0)
			continue;

                std::unordered_map<variableName, VariableAnyValue>&& nodeVariables = graphNode->getNodeVariables();

                this->checkGraphNodeVariables(child.second.get_child("variables"), nodeVariables);
	}
}

std::vector<int> BuildConfigurationCheck::checkGraphNodes(const boost::property_tree::ptree& pt)
{
	std::unordered_map<std::string, PropertyType> properties;
	properties["id"] = PropertyType::InnerType::INT;
	properties["name"] = PropertyType::InnerType::STRING;
	properties["variables"] = PropertyType::InnerType::MAP;

	for (const boost::property_tree::ptree::value_type& child : pt)
		this->checkChildren(child.second, properties);

	std::vector<int> ids = this->checkGraphNodesIds(pt);
	this->checkGraphNodesNamesAndVariables(pt);
	return ids;
}

std::vector<std::pair<int, int>> BuildConfigurationCheck::createGraphNodesConnections(const boost::property_tree::ptree& pt)
{
	std::vector<std::pair<int, int>> connections;
	for (const boost::property_tree::ptree::value_type& child : pt)
		connections.push_back(std::make_pair(child.second.get_child("srcId").get_value<int>(), child.second.get_child("dstId").get_value<int>()));
	return connections;
}

void BuildConfigurationCheck::checkZeroNodeConnection(const std::vector<std::pair<int, int>>& connections)
{
	bool zeroNode = false;
	for (const std::pair<int, int> conn : connections)
	{
		if (conn.first == 0)
		{
			if (zeroNode)
				throw BuildConfigurationCheck::CheckError(std::string("Build configuration: Only one connection to starting node with 0 id can be created."));
			else
				zeroNode = true;
		}
		if (conn.second == 0)
			throw BuildConfigurationCheck::CheckError(std::string("Build configuration: Node with 0 id in graphNodesConnections cannot be a destination node."));
	}

	if (!zeroNode)
		throw BuildConfigurationCheck::CheckError(std::string("Build configuration: Starting node with 0 id is not connected."));
}

void BuildConfigurationCheck::checkNodesExistanceInGraphNodesConnections(const std::vector<std::pair<int, int>>& connections, const std::vector<int>& ids)
{
	std::vector<int> nodesIdsInConnections;
	for (std::pair<int, int> conn : connections)
	{
		nodesIdsInConnections.push_back(conn.first);
		nodesIdsInConnections.push_back(conn.second);
	}
	std::sort(nodesIdsInConnections.begin(), nodesIdsInConnections.end());
	std::vector<int>::iterator last = std::unique(nodesIdsInConnections.begin(), nodesIdsInConnections.end());
	nodesIdsInConnections.erase(last, nodesIdsInConnections.end());

	for (const int nodeId : nodesIdsInConnections)
	{
		if (nodeId != 0)
		{
			std::vector<int>::const_iterator it = std::find(ids.begin(), ids.end(), nodeId);
			if (it == ids.end())
				throw BuildConfigurationCheck::CheckError(std::string("Build configuration: Node with ") + std::to_string(nodeId) +
					std::string(" id in graphNodesConnections is not present in graphNodes."));
		}
	}
}

void BuildConfigurationCheck::searchForDetachedNodesInGraphNodesConnections(const std::vector<std::pair<int, int>>& connections, const std::vector<int>& ids)
{
	std::vector<int> dstNodesIdsInConnections;
	for (std::pair<int, int> conn : connections)
		dstNodesIdsInConnections.push_back(conn.second);

	std::sort(dstNodesIdsInConnections.begin(), dstNodesIdsInConnections.end());
	std::vector<int>::iterator last = std::unique(dstNodesIdsInConnections.begin(), dstNodesIdsInConnections.end());
	dstNodesIdsInConnections.erase(last, dstNodesIdsInConnections.end());

	if (ids.size() != dstNodesIdsInConnections.size())
	{
		for (const int nodeId : ids)
		{
			std::vector<int>::iterator it = std::find(dstNodesIdsInConnections.begin(), dstNodesIdsInConnections.end(), nodeId);
			if (it == dstNodesIdsInConnections.end())
			{
				DISPATCHER_LOG(DispatcherLogType::WARNING, std::string("Build configuration: Node with ") + std::to_string(nodeId) + 
					std::string(" id in graphNodesConnections is not accessible in calculation flow."));
			}
		}
	}
}

void BuildConfigurationCheck::checkFlowDivergence(const std::vector<std::pair<int, int>>& connections, const boost::property_tree::ptree& graphNodesPt)
{
	std::vector<int> srcNodesIdsInConnections;
	for (std::pair<int, int> conn : connections)
		srcNodesIdsInConnections.push_back(conn.first);

	std::sort(srcNodesIdsInConnections.begin(), srcNodesIdsInConnections.end());
	int lastValue = srcNodesIdsInConnections[0];
	for (int i = 1; i < srcNodesIdsInConnections.size(); ++i)
	{
		int currValue = srcNodesIdsInConnections[i];
		if (currValue == lastValue)
		{
			for (const boost::property_tree::ptree::value_type& graphNode : graphNodesPt)
			{
				int id = graphNode.second.get_child("id").get_value<int>();
				if (id == currValue)
				{
					std::string name = graphNode.second.get_child("name").get_value<std::string>();
					if (name.compare("switcher") != 0)
					{
						DISPATCHER_LOG(DispatcherLogType::WARNING, std::string("Build configuration: Node with ") + std::to_string(currValue) +
							std::string(" id in graphNodesConnections creates flow divergence which is allowed only for switcher graph node."));
					}
				}
			}
		}
		lastValue = currValue;
	}
}

void BuildConfigurationCheck::checkGraphNodesConnections(const boost::property_tree::ptree& pt, const std::vector<int>& ids, const boost::property_tree::ptree& graphNodesPt)
{
	std::unordered_map<std::string, PropertyType> properties;
	properties["srcId"] = PropertyType::InnerType::INT;
	properties["dstId"] = PropertyType::InnerType::INT;

	for (const boost::property_tree::ptree::value_type& child : pt)
		this->checkChildren(child.second, properties);

	std::vector<std::pair<int, int>> connections = this->createGraphNodesConnections(pt);
	this->checkZeroNodeConnection(connections);
	this->checkNodesExistanceInGraphNodesConnections(connections, ids);
	this->searchForDetachedNodesInGraphNodesConnections(connections, ids);
	this->checkFlowDivergence(connections, graphNodesPt);
}
