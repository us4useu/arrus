#pragma once

#include "Model/GraphNodesLibrary/GraphNodes/GraphNode.h"
#include "Model/GraphNodesLibrary/GraphNodesFactory.h"

class ConeShapedTransformationGraphNode : public GraphNode {
private:
    static GraphNodesFactoryRegister <ConeShapedTransformationGraphNode> graphNodesFactoryRegister;

public:
    ConeShapedTransformationGraphNode();

    ~ConeShapedTransformationGraphNode();

    void process(cudaStream_t &defaultStream);
};

