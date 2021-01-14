#pragma once

#include "Model/GraphNodesLibrary/GraphNodes/GraphNode.h"
#include "Model/GraphNodesLibrary/GraphNodesFactory.h"
#include "Model/GraphNodesLibrary/GraphNodes/HilbertGraphNode/CudaHilbertGraphNode.cuh"

class HilbertGraphNode : public GraphNode {
private:
    static GraphNodesFactoryRegister <HilbertGraphNode> graphNodesFactoryRegister;
    CudaHilbertGraphNode cudaHilbertGraphNode;
public:
    HilbertGraphNode();

    ~HilbertGraphNode();

    void process(cudaStream_t &defaultStream);
};

