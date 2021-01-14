#include "Model/GraphNodesLibrary/GraphNodes/GraphNode.h"
#include "Utils/DispatcherLogger.h"
#include <boost/algorithm/string.hpp>

GraphNode::GraphNode() {

}

GraphNode::~GraphNode() {
}

void GraphNode::setNodeId(const int nodeId) {
    this->nodeId = nodeId;
}

int GraphNode::getNodeId() {
    return this->nodeId;
}

void GraphNode::setNodeVariables(const std::unordered_map <variableName, VariableAnyValue> &nodeVariables) {
    for(std::pair <variableName, VariableAnyValue> variable : nodeVariables)
        this->setNodeVariable(variable.first, variable.second);
}

std::unordered_map <variableName, VariableAnyValue> GraphNode::getNodeVariables() {
    return this->nodeVariables;
}

VariableAnyValue &GraphNode::getNodeVariable(const variableName name) {
    std::unordered_map<variableName, VariableAnyValue>::iterator it = this->nodeVariables.find(name);
    if(it == this->nodeVariables.end())
        DISPATCHER_LOG(DispatcherLogType::FATAL, std::string("Node with id: ") + std::to_string(this->nodeId) +
                                                 std::string(" doesn't have variable named ") + name);
    return it->second;
}

boost::optional <VariableAnyValue> GraphNode::getOptionalNodeVariable(const variableName name) {
    std::unordered_map<variableName, VariableAnyValue>::iterator it = this->nodeVariables.find(name);
    if(it == this->nodeVariables.end())
        return boost::none;
    return it->second;
}

VariableAnyValue GraphNode::getOptionalNodeVariable(const variableName name, const VariableAnyValue defaultValue) {
    boost::optional <VariableAnyValue> variable = this->getOptionalNodeVariable(name);
    if(variable)
        return variable.get();
    return defaultValue;
}

void GraphNode::setNodeVariable(const variableName &variableNameValue, const VariableAnyValue variableValue) {
    std::vector <std::string> nestedVariablesNames;
    boost::split(nestedVariablesNames, variableNameValue, boost::is_any_of("\\"));
    std::unordered_map <variableName, VariableAnyValue> *currVariablesLevel = &this->nodeVariables;
    for(int i = 1; i < nestedVariablesNames.size(); ++i)
        currVariablesLevel =
            (*currVariablesLevel)[nestedVariablesNames[i - 1]].getValuePtr < std::unordered_map < variableName,
            VariableAnyValue >> ();

    (*currVariablesLevel)[nestedVariablesNames.back()] = variableValue;
}

void GraphNode::addPredecessor(const std::shared_ptr <GraphNode> pred) {
    this->predecessors.push_back(pred);
}

void GraphNode::addSuccessor(const std::shared_ptr <GraphNode> succ) {
    this->successors.push_back(succ);
}

std::vector <std::shared_ptr<GraphNode>> &GraphNode::getSuccessors() {
    return this->successors;
}

DataPtr GraphNode::getOutputData() {
    return this->outputData;
}

void GraphNode::setInputData(const DataPtr &ptr) {
    this->inputData = ptr;
}

void GraphNode::registerCallback(
    const boost::function<void(void *data, int iterationId, int graphNodeId, int dimX, int dimY, int dimZ,
                               int dataType)> callback) {
    DISPATCHER_LOG(DispatcherLogType::WARNING, std::string("Node with id: ") + std::to_string(this->nodeId) +
                                               std::string(" doesn't implement callback function."));
}

void GraphNode::releaseGPUMemory(DataPtr *ptr) {
    if(ptr->getAllocatedDataSize() != 0) {
        CUDA_ASSERT(cudaFree(ptr->getVoidPtr()));
    }
}