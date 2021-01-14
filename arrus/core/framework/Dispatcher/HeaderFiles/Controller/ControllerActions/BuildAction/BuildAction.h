#pragma once

#include <boost/property_tree/ptree.hpp>
#include <unordered_map>

#include "Controller/ControllerActions/ControllerAction.h"
#include "Controller/ControllerEvents/ControllerEvent.h"
#include "Controller/Controller.h"

class BuildAction : public ControllerAction {
private:
    bool parseJson(const std::string &json);

    void parseGraphNodes(const boost::property_tree::ptree &pt, const int graphIndex);

    void parseGraphNodesConnections(const boost::property_tree::ptree &pt, const int graphIndex);

    std::unordered_map <variableName, VariableAnyValue>
    getNodeVariables(const boost::property_tree::ptree &pt, const int graphIndex);

    VariableAnyValue getNodeVariableValue(const boost::property_tree::ptree &pt, const int graphIndex);

    VariableAnyValue parseSingleVariableValue(const boost::property_tree::ptree &pt, const int graphIndex);

    VariableAnyValue parseCompoundVariableValue(const boost::property_tree::ptree &pt, const int graphIndex);

    void recursiveMergeOfNodesVariables(std::unordered_map <variableName, VariableAnyValue> *oldVals,
                                        const std::unordered_map <variableName, VariableAnyValue> &newVals);

public:
    BuildAction(Controller *controller);

    ~BuildAction();

    void performAction(std::shared_ptr <IControllerEvent> controllerEvent);
};

