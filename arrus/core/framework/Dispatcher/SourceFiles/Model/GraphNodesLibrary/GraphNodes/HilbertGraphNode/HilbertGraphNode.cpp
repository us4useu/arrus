#include "Model/GraphNodesLibrary/GraphNodes/HilbertGraphNode/HilbertGraphNode.h"

GraphNodesFactoryRegister <HilbertGraphNode> HilbertGraphNode::graphNodesFactoryRegister("hilbert");

HilbertGraphNode::HilbertGraphNode() {
}


HilbertGraphNode::~HilbertGraphNode() {
    this->releaseGPUMemory(&this->outputData);
}

void HilbertGraphNode::process(cudaStream_t &defaultStream) {
    this->allocGPUMemory<float>(&this->outputData, this->inputData.getDims());

    int batchLength = this->inputData.getDims().y;
    int batchCount = this->inputData.getDims().x;

    this->cudaHilbertGraphNode.hilbertTransform(this->inputData.getPtr<float *>(), this->outputData.getPtr<float *>(),
                                                defaultStream, batchLength, batchCount);

    this->outputData.copyExtraData(this->inputData);
}