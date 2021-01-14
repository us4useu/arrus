#include <cfloat>
#include <cmath>
#include "Model/GraphNodesLibrary/GraphNodes/PwiGraphNode/PwiGraphNode.h"
#include "Utils/DispatcherLogger.h"

GraphNodesFactoryRegister <PwiGraphNode> PwiGraphNode::graphNodesFactoryRegister("pwi");

PwiGraphNode::PwiGraphNode() {
    this->setNodeVariable("width", VariableAnyValue(256));
    this->setNodeVariable("height", VariableAnyValue(512));
    this->setNodeVariable("apodization", VariableAnyValue(std::string("none")));
    std::unordered_map <std::string, VariableAnyValue> pixelMapVariables;
    pixelMapVariables["x"] = VariableAnyValue(-FLT_MAX);
    pixelMapVariables["y"] = VariableAnyValue(-FLT_MAX);
    pixelMapVariables["width"] = VariableAnyValue(-FLT_MAX);
    pixelMapVariables["height"] = VariableAnyValue(-FLT_MAX);
    this->setNodeVariable("pixelMap", VariableAnyValue(pixelMapVariables));
}

PwiGraphNode::~PwiGraphNode() {
    this->releaseGPUMemory(&this->outputData);
    this->releaseGPUMemory(&this->anglesInfo);
    this->releaseGPUMemory(&this->hanningWindow);
}

void PwiGraphNode::sendCurrentAnglesInfoToGPU(const std::vector<float> &angles, const cudaStream_t &defaultStream) {
    std::vector<float> currAnglesInfo(angles.size() * 2);
    for(int i = 0; i < angles.size(); ++i) {
        float currAngleInRadians = angles[i] * 3.14f / 180.0f;
        currAnglesInfo[i * 2] = sinf(currAngleInRadians);
        currAnglesInfo[i * 2 + 1] = cosf(currAngleInRadians);
    }
    this->allocGPUMemory<float>(&this->anglesInfo, (int) currAnglesInfo.size());
    CUDA_ASSERT(
        cudaMemcpyAsync(this->anglesInfo.getVoidPtr(), &currAnglesInfo[0], sizeof(float) * currAnglesInfo.size(),
                        cudaMemcpyHostToDevice, defaultStream));
}

void PwiGraphNode::sendHanningWindowToGPU(const cudaStream_t &defaultStream, const int receiversCount) {
    if(this->cpuHanningWindow.empty()) {
        for(int i = 0; i < receiversCount; ++i) {
            float currCoeff = 0.5f * (1.0f - cosf(2.0f * 3.14f * i / (receiversCount - 1)));
            this->cpuHanningWindow.push_back(currCoeff);
        }
        this->allocGPUMemory<float>(&this->hanningWindow, receiversCount);
        CUDA_ASSERT(cudaMemcpyAsync(this->hanningWindow.getVoidPtr(), &this->cpuHanningWindow[0],
                                    sizeof(float) * receiversCount, cudaMemcpyHostToDevice, defaultStream));
    }
}

PWI_APODIZATION PwiGraphNode::getChosenApodization() {
    std::string apod = this->getNodeVariable("apodization").getValue<std::string>();
    if(apod.compare("none") == 0)
        return PWI_APODIZATION::NONE;
    else if(apod.compare("hann") == 0)
        return PWI_APODIZATION::HANN;
    else if(apod.compare("tas") == 0)
        return PWI_APODIZATION::TAS;

    DISPATCHER_LOG(DispatcherLogType::WARNING,
                   std::string("Unknown apodization type: ") + apod + std::string(". Default none used instead."));

    return PWI_APODIZATION::NONE;
}

CudaPwiPixelMap PwiGraphNode::getPixelMap(float receiverWidth, float areaHeight, float startDepth) {
    CudaPwiPixelMap pixelMap;
    std::unordered_map < std::string,
        VariableAnyValue > pixelMapVariables =
            this->getNodeVariable("pixelMap").getValue < std::unordered_map < std::string, VariableAnyValue >> ();
    float x = pixelMapVariables["x"].getValue<float>();
    pixelMap.x = (x == -FLT_MAX) ? 0.0f : (x + receiverWidth * 0.5f);
    float y = pixelMapVariables["y"].getValue<float>();
    pixelMap.y = (y == -FLT_MAX) ? startDepth : y;

    float width = pixelMapVariables["width"].getValue<float>();
    if(width == -FLT_MAX)
        pixelMap.width = receiverWidth;
    else if(width <= 0.0f) {
        DISPATCHER_LOG(DispatcherLogType::ERROR_,
                       std::string("PwiGraphNode: The width variable in pixel map has value less than zero: ") +
                       std::to_string(width) + std::string(". Default value used instead."));
        pixelMap.width = receiverWidth;
    } else
        pixelMap.width = width;

    float height = pixelMapVariables["height"].getValue<float>();
    if(height == -FLT_MAX)
        pixelMap.height = areaHeight;
    else if(height <= 0.0f) {
        DISPATCHER_LOG(DispatcherLogType::ERROR_,
                       std::string("PwiGraphNode: The height variable in pixel map has value less than zero: ") +
                       std::to_string(height) + std::string(". Default value used instead."));
        pixelMap.height = areaHeight;
    } else
        pixelMap.height = height;

    return pixelMap;
}

void PwiGraphNode::process(cudaStream_t &defaultStream) {
    int width = this->getNodeVariable("width").getValue<int>();
    int height = this->getNodeVariable("height").getValue<int>();
    bool iq = this->inputData.getPtrProperty("iq").getValue<bool>();
    if(iq)
        this->allocGPUMemory<float2>(&this->outputData, Dims(width, height));
    else
        this->allocGPUMemory<float>(&this->outputData, Dims(width, height));

    std::vector<float> angles = this->inputData.getPtrProperty("steeringAngles").getValue < std::vector < float >> ();
    int samplesCount = this->inputData.getPtrProperty("samplesCount").getValue<int>();
    float receiverWidth = this->inputData.getPtrProperty("receiverWidth").getValue<float>();
    float soundVelocity = this->inputData.getPtrProperty("speedOfSound").getValue<float>();
    int receiversCount = this->inputData.getPtrProperty("numReceivers").getValue<int>();
    float samplingFrequency = this->inputData.getPtrProperty("samplingFrequency").getValue<float>();
    float startDepth = this->inputData.getPtrProperty("startDepth").getValue<float>();
    float transmitFrequency =
        this->inputData.getPtrProperty("transmitFrequencies").getValue < std::vector < float >> ()[0];

    float areaHeight = samplesCount * soundVelocity / samplingFrequency * 0.5f;

    this->sendCurrentAnglesInfoToGPU(angles, defaultStream);
    this->sendHanningWindowToGPU(defaultStream, receiversCount);
    PWI_APODIZATION apod = this->getChosenApodization();


    if(iq)
        this->cudaPwiGraphNode.pwiIq(this->inputData.getPtr<float2 *>(), this->outputData.getPtr<float2 *>(),
                                     defaultStream, width, height, (int) angles.size(), soundVelocity, receiverWidth,
                                     receiversCount, samplingFrequency, samplesCount, startDepth,
                                     this->anglesInfo.getPtr<float *>(), this->hanningWindow.getPtr<float *>(),
                                     this->getPixelMap(receiverWidth, areaHeight, startDepth), transmitFrequency, apod);
    else
        this->cudaPwiGraphNode.pwiRf(this->inputData.getPtr<float *>(), this->outputData.getPtr<float *>(),
                                     defaultStream, width, height, (int) angles.size(), soundVelocity, receiverWidth,
                                     receiversCount, samplingFrequency, samplesCount, startDepth,
                                     this->anglesInfo.getPtr<float *>(), this->hanningWindow.getPtr<float *>(),
                                     this->getPixelMap(receiverWidth, areaHeight, startDepth), transmitFrequency, apod);

    this->outputData.copyExtraData(this->inputData);

    if(iq)
        this->outputData.setPtrProperty("iq", VariableAnyValue(true));
    else
        this->outputData.setPtrProperty("iq", VariableAnyValue(false));
}
