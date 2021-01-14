#pragma once

#include "Model/GraphNodesLibrary/GraphNodes/GraphNode.h"
#include "Model/GraphNodesLibrary/GraphNodesFactory.h"
#include "Model/GraphNodesLibrary/GraphNodes/Filter1DGraphNode/Filter1DGraphNode.h"

class QuadratureDemodulationGraphNode : public GraphNode {
private:
    static GraphNodesFactoryRegister <QuadratureDemodulationGraphNode> graphNodesFactoryRegister;
    Filter1DGraphNode filter1DGraphNode;
    DataPtr iqBuffer;

public:
    QuadratureDemodulationGraphNode();

    ~QuadratureDemodulationGraphNode();

    void process(cudaStream_t &defaultStream);
};

