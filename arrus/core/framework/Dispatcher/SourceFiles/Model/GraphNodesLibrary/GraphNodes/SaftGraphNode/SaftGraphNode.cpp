#include "Model/GraphNodesLibrary/GraphNodes/SaftGraphNode/SaftGraphNode.h"

GraphNodesFactoryRegister <SaftGraphNode> SaftGraphNode::graphNodesFactoryRegister("saft");

SaftGraphNode::SaftGraphNode() {

}

SaftGraphNode::~SaftGraphNode() {
    this->releaseGPUMemory(&this->outputData);
}

void SaftGraphNode::process(cudaStream_t &defaultStream) {
    const int signalLen = this->inputData.getDims().x;
    const int apertureLen = this->inputData.getDims().y;

    this->allocGPUMemory<float>(&this->outputData, Dims(apertureLen, signalLen));

    std::vector<float> angles = this->inputData.getPtrProperty("steeringAngles").getValue < std::vector < float >> ();
    const float soundVelocity = this->inputData.getPtrProperty("speedOfSound").getValue<float>();
    const float fs = this->inputData.getPtrProperty("samplingFrequency").getValue<float>();
    const float pitch = this->inputData.getPtrProperty("pitch").getValue<float>();
    const float t0 = this->inputData.getPtrProperty("startDepth").getValue<float>();

    cudaSaftGraphNode.saft(this->inputData.getPtr<float *>(),
                           this->outputData.getPtr<float *>(),
                           apertureLen,
                           signalLen,
                           angles,
                           t0,
                           soundVelocity,
                           fs,
                           pitch,
                           defaultStream);

    this->outputData.copyExtraData(this->inputData);
}