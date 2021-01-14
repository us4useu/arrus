#include "Model/GraphNodesLibrary/GraphNodes/TgcGraphNode/TgcGraphNode.h"
#include "Model/GraphNodesLibrary/GraphNodes/TgcGraphNode/CudaTgcGraphNode.cuh"
#include "Utils/DispatcherLogger.h"

GraphNodesFactoryRegister <TgcGraphNode> TgcGraphNode::graphNodesFactoryRegister("tgc");

TgcGraphNode::TgcGraphNode() {
    this->setNodeVariable("func", VariableAnyValue(std::string("lin")));
    this->setNodeVariable("A", VariableAnyValue(0.0f));
    this->setNodeVariable("B", VariableAnyValue(0.0f));
    this->setNodeVariable("C", VariableAnyValue(0.0f));
}

TgcGraphNode::~TgcGraphNode() {
    this->releaseGPUMemory(&this->outputData);
}

TGC_FUNC_TYPE TgcGraphNode::getChosenTgcFunctionType() {
    std::string tgcFuncType = this->getNodeVariable("func").getValue<std::string>();
    if(tgcFuncType.compare("lin") == 0)
        return TGC_FUNC_TYPE::LIN;
    else if(tgcFuncType.compare("exp") == 0)
        return TGC_FUNC_TYPE::EXP;

    DISPATCHER_LOG(DispatcherLogType::WARNING,
                   std::string("Unknown tgc function type. Default linear function used instead."));
    return TGC_FUNC_TYPE::LIN;
}

std::unordered_map<std::string, float> TgcGraphNode::getFunctionParams() {
    std::unordered_map<std::string, float> params;
    params["A"] = this->getNodeVariable("A").getValue<float>();
    params["B"] = this->getNodeVariable("B").getValue<float>();
    params["C"] = this->getNodeVariable("C").getValue<float>();
    return params;
}

void TgcGraphNode::process(cudaStream_t &defaultStream) {
    this->allocGPUMemory<float>(&this->outputData, this->inputData.getDims());

    TGC_FUNC_TYPE tgcFuncType = this->getChosenTgcFunctionType();
    std::unordered_map<std::string, float> params = this->getFunctionParams();

    int samplesCount = this->inputData.getPtrProperty("samplesCount").getValue<int>();
    float soundVelocity = this->inputData.getPtrProperty("speedOfSound").getValue<float>();
    float samplingFrequency = this->inputData.getPtrProperty("samplingFrequency").getValue<float>();
    float areaHeight = samplesCount * soundVelocity / samplingFrequency * 0.5f;

    CudaTgcGraphNode::tgc(this->inputData.getPtr<float *>(), this->outputData.getPtr<float *>(), defaultStream,
                          this->inputData.getDims().x,
                          this->inputData.getDims().y, areaHeight, params, tgcFuncType);

    this->outputData.copyExtraData(this->inputData);
}