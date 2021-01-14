#pragma once

#include "Model/GraphNodesLibrary/GraphNodes/GraphNode.h"
#include "Model/GraphNodesLibrary/GraphNodesFactory.h"
#include "Model/GraphNodesLibrary/GraphNodes/PwiGraphNode/CudaPwiGraphNode.cuh"

class PwiGraphNode : public GraphNode {
private:
    static GraphNodesFactoryRegister <PwiGraphNode> graphNodesFactoryRegister;
    DataPtr anglesInfo;
    DataPtr hanningWindow;
    CudaPwiGraphNode cudaPwiGraphNode;
    std::vector<float> cpuHanningWindow;

    void sendCurrentAnglesInfoToGPU(const std::vector<float> &angles, const cudaStream_t &defaultStream);

    void sendHanningWindowToGPU(const cudaStream_t &defaultStream, const int receiversCount);

    PWI_APODIZATION getChosenApodization();

    CudaPwiPixelMap getPixelMap(float receiverWidth, float areaHeight, float startDepth);

public:
    PwiGraphNode();

    ~PwiGraphNode();

    void process(cudaStream_t &defaultStream);
};

