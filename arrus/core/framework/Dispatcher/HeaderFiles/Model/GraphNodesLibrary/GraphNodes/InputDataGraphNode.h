#pragma once

#include "Model/GraphNodesLibrary/GraphNodes/GraphNode.h"
#include "Model/GraphNodesLibrary/GraphNodesFactory.h"
#include "Model/IntelligentBuffer.h"

class InternalMemcpyCallbackData {
private:
    IntelligentBuffer *intelligentBuffer;
    elementIndex index;
public:
    InternalMemcpyCallbackData(IntelligentBuffer *intelligentBuffer, elementIndex index) : intelligentBuffer(
        intelligentBuffer), index(index) {};

    void process() {
        this->intelligentBuffer->setDataEmpty(this->index);
    }
};

class InputDataGraphNode : public GraphNode {
private:
    static GraphNodesFactoryRegister <InputDataGraphNode> graphNodesFactoryRegister;
    IntelligentBuffer *intelligentBuffer;
    cudaStream_t dataStream;
    DataPtr overlapData;
public:
    InputDataGraphNode();

    ~InputDataGraphNode();

    void process(cudaStream_t &defaultStream);

    void setIntelligentBuffer(IntelligentBuffer *intelligentBuffer);

    static void CUDART_CB
    internalMemcpyCallback(cudaStream_t
    stream,
    cudaError_t status,
    void *callData
    );
};

