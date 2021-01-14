#pragma once

#include "Model/GraphNodesLibrary/GraphNodes/GraphNode.h"
#include "Model/GraphNodesLibrary/GraphNodesFactory.h"
#include "Model/GraphNodesLibrary/GraphNodes/BModeGraphNode/CudaBModeGraphNode.cuh"

class BModeGraphNode : public GraphNode {
private:
    static GraphNodesFactoryRegister <BModeGraphNode> graphNodesFactoryRegister;
    CudaBModeGraphNode cudaBModeGraphNode;
    DataPtr complexModulusData;
public:
    BModeGraphNode();

    ~BModeGraphNode();

    void process(cudaStream_t &defaultStream);
};

