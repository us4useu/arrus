#pragma once

#include "Model/GraphNodesLibrary/GraphNodes/GraphNode.h"
#include "Model/GraphNodesLibrary/GraphNodesFactory.h"

class OutputDataGraphNode : public GraphNode {
private:
    static GraphNodesFactoryRegister <OutputDataGraphNode> graphNodesFactoryRegister;
    DataPtr hostData, inputDataCopy;
    cudaStream_t dataStream;

    void releaseMemory();

    void allocMemory();

public:
    OutputDataGraphNode();

    ~OutputDataGraphNode();

    void process(cudaStream_t &defaultStream);

    void registerCallback(
        const boost::function<void(void *data, int iterationId, int graphNodeId, int dimX, int dimY, int dimZ,
                                   int dataType)> callbackFunc);

    static void CUDART_CB
    callback(cudaStream_t
    stream,
    cudaError_t status,
    void *userData
    );
};

