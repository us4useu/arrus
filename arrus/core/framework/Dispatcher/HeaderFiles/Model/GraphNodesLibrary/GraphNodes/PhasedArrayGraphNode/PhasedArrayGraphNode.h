#pragma once

#include "Model/GraphNodesLibrary/GraphNodes/GraphNode.h"
#include "Model/GraphNodesLibrary/GraphNodesFactory.h"

class PhasedArrayGraphNode : public GraphNode {
private:
    static GraphNodesFactoryRegister <PhasedArrayGraphNode> graphNodesFactoryRegister;
    DataPtr anglesInfo, focusesInfo;

public:
    PhasedArrayGraphNode();

    ~PhasedArrayGraphNode();

    void process(cudaStream_t &defaultStream);
};

