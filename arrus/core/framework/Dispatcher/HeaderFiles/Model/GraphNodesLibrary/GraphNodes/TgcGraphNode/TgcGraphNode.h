#pragma once

#include "Model/GraphNodesLibrary/GraphNodes/GraphNode.h"
#include "Model/GraphNodesLibrary/GraphNodesFactory.h"
#include "Model/GraphNodesLibrary/GraphNodes/TgcGraphNode/CudaTgcGraphNode.cuh"

class TgcGraphNode : public GraphNode {
private:
    static GraphNodesFactoryRegister <TgcGraphNode> graphNodesFactoryRegister;

    TGC_FUNC_TYPE getChosenTgcFunctionType();

    std::unordered_map<std::string, float> getFunctionParams();

public:
    TgcGraphNode();

    ~TgcGraphNode();

    void process(cudaStream_t &defaultStream);
};

