#include "Model/GraphNodesLibrary/GraphNodes/BModeGraphNode/BModeGraphNode.h"

GraphNodesFactoryRegister <BModeGraphNode> BModeGraphNode::graphNodesFactoryRegister("bMode");

BModeGraphNode::BModeGraphNode() {
    this->setNodeVariable("minDBLimit", VariableAnyValue(-60.0f));
    this->setNodeVariable("maxDBLimit", VariableAnyValue(-1.0f));
    this->setNodeVariable("maxDataValue", VariableAnyValue(FLT_MAX));
}

BModeGraphNode::~BModeGraphNode() {
    this->releaseGPUMemory(&this->outputData);
}

void BModeGraphNode::process(cudaStream_t &defaultStream) {
    this->allocGPUMemory<float>(&this->complexModulusData, this->inputData.getDims());
    this->allocGPUMemory<float>(&this->outputData, this->inputData.getDims());

    float minDBLimit = this->getNodeVariable("minDBLimit").getValue<float>();
    float maxDBLimit = this->getNodeVariable("maxDBLimit").getValue<float>();
    float maxDataValue = this->getNodeVariable("maxDataValue").getValue<float>();
    int dataCount = this->inputData.getDims().flatten();

    bool iq = this->inputData.getPtrProperty("iq").getValue<bool>();

    if(iq)
        cudaBModeGraphNode.convertToBModeIq(this->inputData.getPtr<float2 *>(),
                                            this->complexModulusData.getPtr<float *>(),
                                            this->outputData.getPtr<float *>(),
                                            defaultStream, minDBLimit, maxDBLimit, dataCount, maxDataValue);
    else
        cudaBModeGraphNode.convertToBMode(this->inputData.getPtr<float *>(), this->outputData.getPtr<float *>(),
                                          defaultStream, minDBLimit, maxDBLimit, dataCount, maxDataValue);
    this->outputData.copyExtraData(this->inputData);
}