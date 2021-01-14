#pragma once

#include "Model/GraphNodesLibrary/GraphNodes/GraphNode.h"
#include "Model/GraphNodesLibrary/GraphNodesFactory.h"
#include "Model/GraphNodesLibrary/GraphNodes/StaGraphNode/CudaStaGraphNode.cuh"

class StaGraphNode : public GraphNode {
private:
    static GraphNodesFactoryRegister <StaGraphNode> graphNodesFactoryRegister;
    DataPtr transmittersInfo;
    DataPtr hanningWindow;
    CudaStaGraphNode cudaStaGraphNode;
    std::vector<float> cpuHanningWindow;

    void sendHanningWindowToGPU(const cudaStream_t &defaultStream, const int receiversCount);

    STA_APODIZATION getChosenApodization();

    CudaStaPixelMap getPixelMap(float receiverWidth, float areaHeight, float startDepth);

public:
    StaGraphNode();

    ~StaGraphNode();

    void process(cudaStream_t &defaultStream);
};
