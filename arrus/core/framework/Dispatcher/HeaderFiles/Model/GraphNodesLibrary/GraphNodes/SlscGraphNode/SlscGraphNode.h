#pragma once

#include "Model/GraphNodesLibrary/GraphNodes/GraphNode.h"
#include "Model/GraphNodesLibrary/GraphNodesFactory.h"
#include "Model/GraphNodesLibrary/GraphNodes/SlscGraphNode/CudaSlscGraphNode.cuh"

class SlscGraphNode : public GraphNode {
private:
    static GraphNodesFactoryRegister <SlscGraphNode> graphNodesFactoryRegister;
    DataPtr transmittersInfo;
    CudaSlscGraphNode cudaSlscGraphNode;

    CudaSlscPixelMap getPixelMap(float receiverWidth, float areaHeight, float startDepth);

public:
    SlscGraphNode();

    ~SlscGraphNode();

    void process(cudaStream_t &defaultStream);
};
