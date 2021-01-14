#pragma once

#include "Model/GraphNodesLibrary/GraphNodes/GraphNode.h"
#include "Model/GraphNodesLibrary/GraphNodesFactory.h"
#include "Model/GraphNodesLibrary/GraphNodes/SaftGraphNode/CudaSaftGraphNode.cuh"

class SaftGraphNode : public GraphNode {
private:
    static GraphNodesFactoryRegister <SaftGraphNode> graphNodesFactoryRegister;
    CudaSaftGraphNode cudaSaftGraphNode;
public:
    SaftGraphNode();

    ~SaftGraphNode();

    void process(cudaStream_t &defaultStream);
};
