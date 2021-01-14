#pragma once

#include "Model/GraphNodesLibrary/GraphNodes/GraphNode.h"
#include "Model/GraphNodesLibrary/GraphNodesFactory.h"

class SwitcherGraphNode : public GraphNode {
private:
    static GraphNodesFactoryRegister <SwitcherGraphNode> graphNodesFactoryRegister;

    void pushSuccessorToFront(const int successorNodeId);

    std::vector<std::shared_ptr < GraphNode>>::

    iterator findSuccessorWithNodeId(const int successorNodeId);

public:
    SwitcherGraphNode();

    ~SwitcherGraphNode();

    void process(cudaStream_t &defaultStream);
};

