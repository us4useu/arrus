#pragma once

#include "Model/GraphNodesLibrary/GraphNodes/GraphNode.h"
#include "Model/GraphNodesLibrary/GraphNodesFactory.h"
#include "Model/GraphNodesLibrary/GraphNodes/SaftZeroGraphNode/CudaSaftZeroGraphNode.cuh"

class SaftZeroGraphNode : public GraphNode {
private:
    static GraphNodesFactoryRegister <SaftZeroGraphNode> graphNodesFactoryRegister;
    CudaSaftZeroGraphNode cudaSaftZeroGraphNode;
public:
    SaftZeroGraphNode();

    ~SaftZeroGraphNode();

    void process(cudaStream_t &defaultStream);
};

