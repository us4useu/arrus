#pragma once

#include <unordered_map>
#include <memory>
#include "Model/VariableAnyValue.h"
#include "Model/DataPtr.h"
#include "boost/function.hpp"
#include "boost/optional.hpp"
#include <cuda_runtime.h>
#include "Model/GraphNodesLibrary/GraphNodes/CudaAssert.cuh"

typedef std::string variableName;

class GraphNode {
protected:
    std::unordered_map <variableName, VariableAnyValue> nodeVariables;
    std::vector <std::shared_ptr<GraphNode>> predecessors;
    std::vector <std::shared_ptr<GraphNode>> successors;
    DataPtr inputData, outputData;
    boost::function<void(void *data, int iterationId, int graphNodeId, int dimX, int dimY, int dimZ,
                         int dataType)> callbackFunc;
    int nodeId;

    template<typename T>
    void allocGPUMemory(DataPtr *ptr, const Dims dims) {
        int dataSize = dims.flatten() * sizeof(T);
        if(ptr->getAllocatedDataSize() < dataSize) {
            this->releaseGPUMemory(ptr);

            T *data;
            CUDA_ASSERT(cudaMalloc((void **) &data, dataSize));
            *ptr = DataPtr(data, dims);
        } else
            ptr->setDims(dims);
    }

    void releaseGPUMemory(DataPtr *ptr);

public:
    GraphNode();

    virtual ~GraphNode();

    std::unordered_map <variableName, VariableAnyValue> getNodeVariables();

    void setNodeVariables(const std::unordered_map <variableName, VariableAnyValue> &nodeVariables);

    VariableAnyValue &getNodeVariable(const variableName variableName);

    VariableAnyValue getOptionalNodeVariable(const variableName variableName, const VariableAnyValue defaultValue);

    boost::optional <VariableAnyValue> getOptionalNodeVariable(const variableName variableName);

    void setNodeVariable(const variableName &variableName, const VariableAnyValue variableValue);

    void addPredecessor(const std::shared_ptr <GraphNode> pred);

    void addSuccessor(const std::shared_ptr <GraphNode> succ);

    std::vector <std::shared_ptr<GraphNode>> &getSuccessors();

    DataPtr getOutputData();

    void setInputData(const DataPtr &ptr);

    void setNodeId(const int nodeId);

    int getNodeId();

    virtual void process(cudaStream_t &defaultStream) = 0;

    virtual void registerCallback(
        const boost::function<void(void *data, int iterationId, int graphNodeId, int dimX, int dimY, int dimZ,
                                   int dataType)> callbackFunc);
};

