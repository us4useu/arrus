#include <cfloat>
#include "Model/GraphNodesLibrary/GraphNodes/SlscGraphNode/SlscGraphNode.h"
#include "Utils/DispatcherLogger.h"

GraphNodesFactoryRegister <SlscGraphNode> SlscGraphNode::graphNodesFactoryRegister("slsc");

SlscGraphNode::SlscGraphNode() {
    this->setNodeVariable("width", VariableAnyValue(256));
    this->setNodeVariable("height", VariableAnyValue(512));
    this->setNodeVariable("focusing", VariableAnyValue(false));
    this->setNodeVariable("lags", VariableAnyValue(40));
    this->setNodeVariable("offset", VariableAnyValue(2));
    std::unordered_map <std::string, VariableAnyValue> pixelMapVariables;
    pixelMapVariables["x"] = VariableAnyValue(-FLT_MAX);
    pixelMapVariables["y"] = VariableAnyValue(-FLT_MAX);
    pixelMapVariables["width"] = VariableAnyValue(-FLT_MAX);
    pixelMapVariables["height"] = VariableAnyValue(-FLT_MAX);
    this->setNodeVariable("pixelMap", VariableAnyValue(pixelMapVariables));
}

SlscGraphNode::~SlscGraphNode() {
    this->releaseGPUMemory(&this->outputData);
    this->releaseGPUMemory(&this->transmittersInfo);
}

CudaSlscPixelMap SlscGraphNode::getPixelMap(float receiverWidth, float areaHeight, float startDepth) {
    CudaSlscPixelMap pixelMap;
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
                       std::string("SlscGraphNode: The width variable in pixel map has value less than zero: ") +
                       std::to_string(width) + std::string(". Default value used instead."));
        pixelMap.width = receiverWidth;
    } else
        pixelMap.width = width;

    float height = pixelMapVariables["height"].getValue<float>();
    if(height == -FLT_MAX)
        pixelMap.height = areaHeight;
    else if(height <= 0.0f) {
        DISPATCHER_LOG(DispatcherLogType::ERROR_,
                       std::string("SlscGraphNode: The height variable in pixel map has value less than zero: ") +
                       std::to_string(height) + std::string(". Default value used instead."));
        pixelMap.height = areaHeight;
    } else
        pixelMap.height = height;

    return pixelMap;
}

void SlscGraphNode::process(cudaStream_t &defaultStream) {
    int width = this->getNodeVariable("width").getValue<int>();
    int height = this->getNodeVariable("height").getValue<int>();
    this->allocGPUMemory<float>(&this->outputData, Dims(width, height));

    int samplesCount = this->inputData.getPtrProperty("samplesCount").getValue<int>();
    float receiverWidth = this->inputData.getPtrProperty("receiverWidth").getValue<float>();
    float soundVelocity = this->inputData.getPtrProperty("speedOfSound").getValue<float>();
    int receiversCount = this->inputData.getPtrProperty("numReceivers").getValue<int>();
    float samplingFrequency = this->inputData.getPtrProperty("samplingFrequency").getValue<float>();
    float transmitFrequency =
        this->inputData.getPtrProperty("transmitFrequencies").getValue < std::vector < float >> ()[0];
    float startDepth = this->inputData.getPtrProperty("startDepth").getValue<float>();
    std::vector<int> originTransmitters =
        this->inputData.getPtrProperty("originTransmitters").getValue < std::vector < int >> ();

    float areaHeight = samplesCount * soundVelocity / samplingFrequency * 0.5f;

    bool focusing = this->getNodeVariable("focusing").getValue<bool>();
    int lags = this->getNodeVariable("lags").getValue<int>();
    int offset = this->getNodeVariable("offset").getValue<int>();
    bool iq = this->inputData.getPtrProperty("iq").getValue<bool>();

    if(!focusing) {
        this->allocGPUMemory<int>(&this->transmittersInfo, (int) originTransmitters.size());
        CUDA_ASSERT(cudaMemcpyAsync(this->transmittersInfo.getVoidPtr(), &originTransmitters[0],
                                    (int) originTransmitters.size() * sizeof(int), cudaMemcpyHostToDevice,
                                    defaultStream));

        if(iq) {
            cudaSlscGraphNode.slscIq(this->inputData.getPtr<float2 *>(), this->outputData.getPtr<float *>(),
                                     defaultStream, width, height, areaHeight, receiversCount,
                                     samplesCount, receiverWidth, soundVelocity, samplingFrequency, startDepth,
                                     this->transmittersInfo.getPtr<int *>(), (int) originTransmitters.size(),
                                     lags, transmitFrequency, this->getPixelMap(receiverWidth, areaHeight, startDepth));
        } else {
            cudaSlscGraphNode.slscRf(this->inputData.getPtr<float *>(), this->outputData.getPtr<float *>(),
                                     defaultStream, width, height, areaHeight, receiversCount,
                                     samplesCount, receiverWidth, soundVelocity, samplingFrequency, startDepth,
                                     this->transmittersInfo.getPtr<int *>(), (int) originTransmitters.size(),
                                     lags, offset, this->getPixelMap(receiverWidth, areaHeight, startDepth));
        }
    } else {
        if(originTransmitters.size() != width) {
            DISPATCHER_LOG(DispatcherLogType::ERROR_, std::string(
                "Number of acquisitions must be equal to image width in slsc with focusing option enabled."));
            return;
        }
        if(iq) {
            cudaSlscGraphNode.slscWithFocusingIq(this->inputData.getPtr<float2 *>(), this->outputData.getPtr<float *>(),
                                                 defaultStream, width, height, areaHeight, receiversCount,
                                                 samplesCount, receiverWidth, soundVelocity, samplingFrequency,
                                                 startDepth, lags, transmitFrequency);
        } else {
            cudaSlscGraphNode.slscWithFocusingRf(this->inputData.getPtr<float *>(), this->outputData.getPtr<float *>(),
                                                 defaultStream, width, height, areaHeight, receiversCount,
                                                 samplesCount, receiverWidth, soundVelocity, samplingFrequency,
                                                 startDepth, lags, offset);
        }
    }

    this->outputData.copyExtraData(this->inputData);
    this->outputData.setPtrProperty("iq", VariableAnyValue(false));
}
