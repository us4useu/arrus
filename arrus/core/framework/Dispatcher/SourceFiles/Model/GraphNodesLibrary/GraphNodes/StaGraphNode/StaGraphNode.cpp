#include <cfloat>
#include <cmath>
#include "Model/GraphNodesLibrary/GraphNodes/StaGraphNode/StaGraphNode.h"
#include "Utils/DispatcherLogger.h"

GraphNodesFactoryRegister <StaGraphNode> StaGraphNode::graphNodesFactoryRegister("sta");

StaGraphNode::StaGraphNode() {
    this->setNodeVariable("width", VariableAnyValue(256));
    this->setNodeVariable("height", VariableAnyValue(512));
    this->setNodeVariable("focusing", VariableAnyValue(false));
    this->setNodeVariable("apodization", VariableAnyValue(std::string("none")));
    std::unordered_map <std::string, VariableAnyValue> pixelMapVariables;
    pixelMapVariables["x"] = VariableAnyValue(-FLT_MAX);
    pixelMapVariables["y"] = VariableAnyValue(-FLT_MAX);
    pixelMapVariables["width"] = VariableAnyValue(-FLT_MAX);
    pixelMapVariables["height"] = VariableAnyValue(-FLT_MAX);
    this->setNodeVariable("pixelMap", VariableAnyValue(pixelMapVariables));
}

StaGraphNode::~StaGraphNode() {
    this->releaseGPUMemory(&this->outputData);
    this->releaseGPUMemory(&this->transmittersInfo);
}

void StaGraphNode::sendHanningWindowToGPU(const cudaStream_t &defaultStream, const int receiversCount) {
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

STA_APODIZATION StaGraphNode::getChosenApodization() {
    std::string apod = this->getNodeVariable("apodization").getValue<std::string>();
    if(apod.compare("none") == 0)
        return STA_APODIZATION::NONE;
    else if(apod.compare("hann") == 0)
        return STA_APODIZATION::HANN;
    else if(apod.compare("tas") == 0)
        return STA_APODIZATION::TAS;

    DISPATCHER_LOG(DispatcherLogType::WARNING,
                   std::string("Unknown apodization type: ") + apod + std::string(". Default none used instead."));

    return STA_APODIZATION::NONE;
}

CudaStaPixelMap StaGraphNode::getPixelMap(float receiverWidth, float areaHeight, float startDepth) {
    CudaStaPixelMap pixelMap;
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
                       std::string("StaGraphNode: The width variable in pixel map has value less than zero: ") +
                       std::to_string(width) + std::string(". Default value used instead."));
        pixelMap.width = receiverWidth;
    } else
        pixelMap.width = width;

    float height = pixelMapVariables["height"].getValue<float>();
    if(height == -FLT_MAX)
        pixelMap.height = areaHeight;
    else if(height <= 0.0f) {
        DISPATCHER_LOG(DispatcherLogType::ERROR_,
                       std::string("StaGraphNode: The height variable in pixel map has value less than zero: ") +
                       std::to_string(height) + std::string(". Default value used instead."));
        pixelMap.height = areaHeight;
    } else
        pixelMap.height = height;

    return pixelMap;
}

void StaGraphNode::process(cudaStream_t &defaultStream) {
    int width = this->getNodeVariable("width").getValue<int>();
    int height = this->getNodeVariable("height").getValue<int>();
    this->allocGPUMemory<float>(&this->outputData, Dims(width, height));

    int samplesCount = this->inputData.getPtrProperty("samplesCount").getValue<int>();
    float receiverWidth = this->inputData.getPtrProperty("receiverWidth").getValue<float>();
    float soundVelocity = this->inputData.getPtrProperty("speedOfSound").getValue<float>();
    int receiversCount = this->inputData.getPtrProperty("numReceivers").getValue<int>();
    float samplingFrequency = this->inputData.getPtrProperty("samplingFrequency").getValue<float>();
    float startDepth = this->inputData.getPtrProperty("startDepth").getValue<float>();
    std::vector<int> originTransmitters =
        this->inputData.getPtrProperty("originTransmitters").getValue < std::vector < int >> ();
    std::vector<int> transmitApertures =
        this->inputData.getPtrProperty("transmitApertures").getValue < std::vector < int >> ();
    float transmitFrequency =
        this->inputData.getPtrProperty("transmitFrequencies").getValue < std::vector < float >> ()[0];

    float areaHeight = samplesCount * soundVelocity / samplingFrequency * 0.5f;

    bool focusing = this->getNodeVariable("focusing").getValue<bool>();
    this->sendHanningWindowToGPU(defaultStream, receiversCount);
    STA_APODIZATION apod = this->getChosenApodization();
    bool iq = this->inputData.getPtrProperty("iq").getValue<bool>();

    if(!focusing) {
        this->allocGPUMemory<float>(&this->transmittersInfo, (int) originTransmitters.size());

        // convert origin transmitters to aperture center in case when transmit aperture is different than 1
        std::vector<float> originRealTransmitters(originTransmitters.size());
        for(int i = 0; i < originTransmitters.size(); ++i)
            originRealTransmitters[i] = (transmitApertures[i] - 1) / 2.0f + (float) originTransmitters[i];

        CUDA_ASSERT(cudaMemcpyAsync(this->transmittersInfo.getVoidPtr(), &originRealTransmitters[0],
                                    (int) originRealTransmitters.size() * sizeof(float), cudaMemcpyHostToDevice,
                                    defaultStream));

        if(iq)
            this->cudaStaGraphNode.staIq(this->inputData.getPtr<float2 *>(), this->outputData.getPtr<float *>(),
                                         defaultStream, width, height, receiversCount, samplesCount,
                                         receiverWidth, soundVelocity, samplingFrequency, startDepth,
                                         this->transmittersInfo.getPtr<float *>(), (int) originRealTransmitters.size(),
                                         this->hanningWindow.getPtr<float *>(),
                                         transmitFrequency, apod,
                                         this->getPixelMap(receiverWidth, areaHeight, startDepth));
        else
            this->cudaStaGraphNode.staRf(this->inputData.getPtr<float *>(), this->outputData.getPtr<float *>(),
                                         defaultStream, width, height, receiversCount, samplesCount,
                                         receiverWidth, soundVelocity, samplingFrequency, startDepth,
                                         this->transmittersInfo.getPtr<float *>(), (int) originRealTransmitters.size(),
                                         this->hanningWindow.getPtr<float *>(),
                                         transmitFrequency, apod,
                                         this->getPixelMap(receiverWidth, areaHeight, startDepth));
    } else {
        if(originTransmitters.size() != width) {
            DISPATCHER_LOG(DispatcherLogType::ERROR_, std::string(
                "Number of acquisitions must be equal to image width in sta with focusing option enabled. Parameters are set to: ") +
                                                      std::string("image width: ") + std::to_string(width) +
                                                      std::string(", number of acquisitions: ") +
                                                      std::to_string((int) originTransmitters.size()));
            return;
        }

        if(iq)
            this->cudaStaGraphNode.staIqWithFocusing(this->inputData.getPtr<float2 *>(),
                                                     this->outputData.getPtr<float *>(), defaultStream, width, height,
                                                     areaHeight, receiversCount,
                                                     samplesCount, receiverWidth, soundVelocity, samplingFrequency,
                                                     startDepth, this->hanningWindow.getPtr<float *>(),
                                                     transmitFrequency, apod);
        else
            this->cudaStaGraphNode.staRfWithFocusing(this->inputData.getPtr<float *>(),
                                                     this->outputData.getPtr<float *>(), defaultStream, width, height,
                                                     areaHeight, receiversCount,
                                                     samplesCount, receiverWidth, soundVelocity, samplingFrequency,
                                                     startDepth, this->hanningWindow.getPtr<float *>(),
                                                     transmitFrequency, apod);
    }

    this->outputData.copyExtraData(this->inputData);
    this->outputData.setPtrProperty("iq", VariableAnyValue(false));
}
