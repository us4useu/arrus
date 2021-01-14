#pragma once

#include <boost/property_tree/ptree.hpp>
#include <unordered_map>
#include "Controller/ControllerActions/BuildAction/PropertyType.h"
#include "Model/GraphNodesLibrary/GraphNodesFactory.h"

class BuildConfigurationCheck {
private:
    class CheckError {
    public:
        CheckError(const std::string &errorMessage) : errorMessage(errorMessage) {};
        std::string errorMessage;
    };

    std::string jsonConfiguration;
    boost::property_tree::ptree parsedJsonConfiguration;

    void parseJsonConfiguration();

    void validateHALJsonConfiguration();

    std::vector<int> checkGraphNodes(const boost::property_tree::ptree &pt);

    void checkGraphNodesConnections(const boost::property_tree::ptree &pt, const std::vector<int> &ids,
                                    const boost::property_tree::ptree &graphNodesPt);

    void
    checkChildren(const boost::property_tree::ptree &parent, std::unordered_map <std::string, PropertyType> properties);

    std::vector<int> checkGraphNodesIds(const boost::property_tree::ptree &parent);

    void checkGraphNodesNamesAndVariables(const boost::property_tree::ptree &parent);

    void checkGraphNodeVariables(const boost::property_tree::ptree &pt,
                                 std::unordered_map <variableName, VariableAnyValue> &nodeVariables);

    std::vector <std::pair<int, int>> createGraphNodesConnections(const boost::property_tree::ptree &pt);

    void checkZeroNodeConnection(const std::vector <std::pair<int, int>> &connections);

    void checkNodesExistanceInGraphNodesConnections(const std::vector <std::pair<int, int>> &connections,
                                                    const std::vector<int> &ids);

    void searchForDetachedNodesInGraphNodesConnections(const std::vector <std::pair<int, int>> &connections,
                                                       const std::vector<int> &ids);

    void checkFlowDivergence(const std::vector <std::pair<int, int>> &connections,
                             const boost::property_tree::ptree &graphNodesPt);

public:
    BuildConfigurationCheck(const std::string &jsonConfiguration);

    bool performCheck();

    const boost::property_tree::ptree &getParsedJson();
};

