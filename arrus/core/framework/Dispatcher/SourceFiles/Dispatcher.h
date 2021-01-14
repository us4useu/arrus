#pragma once

#include <limits>

#include "../HeaderFiles/DispatcherInterface.h"
#include "../HeaderFiles/Controller/Controller.h"
#include "../HeaderFiles/Controller/ControllerEvents/SetUserInputDataEvent.h"
#include "../HeaderFiles/Controller/ControllerEvents/UpdateGraphNodeParameterEvent.h"

class Dispatcher : public DispatcherInterface {
private:
    Controller *controller;

    template<typename T>
    void
    internalSetInputData(T *dataPtr, const int numberOfBatches, const unsigned int dimX, const unsigned int dimY = 1,
                         const unsigned int dimZ = 1) {
        this->controller->sendControllerEvent(std::shared_ptr<SetUserInputDataEvent<T>>(
            new SetUserInputDataEvent<T>(dataPtr, numberOfBatches, Dims(dimX, dimY, dimZ))));
        this->controller->synchronize();
    }

    template<typename T>
    void
    internalUpdateGraphNodeParameterAsync(const int nodeId, const std::string &parameterName, const T &parameterValue) {
        this->controller->sendControllerEvent(std::shared_ptr<UpdateGraphNodeParameterEvent>(
            new UpdateGraphNodeParameterEvent(nodeId, parameterName, VariableAnyValue(parameterValue))));
    }

public:
    DISPATCHERDLL_API Dispatcher();

    DISPATCHERDLL_API ~Dispatcher();

    void DISPATCHERDLL_API

    startAsync();

    void DISPATCHERDLL_API

    startOnce();

    void DISPATCHERDLL_API

    stop();

    void DISPATCHERDLL_API

    kill();

    /** Builds execution graph using JSON string. */
    bool DISPATCHERDLL_API

    build(const std::string &json, const bool cascadeMode = false);

    void DISPATCHERDLL_API

    buildAsync(const std::string &json, const bool cascadeMode = false);

    /** Binds callback function to particular node in execution graph. */
    void DISPATCHERDLL_API

    registerCallback(const int id,
                     void(*callback)(void *data, int iterationId, int graphNodeId, int dimX, int dimY, int dimZ,
                                     int dataType));

    void DISPATCHERDLL_API

    setInputData(short *dataPtr, const int numberOfBatches, const unsigned int dimX, const unsigned int dimY = 1,
                 const unsigned int dimZ = 1);

    void DISPATCHERDLL_API

    setInputData(int *dataPtr, const int numberOfBatches, const unsigned int dimX, const unsigned int dimY = 1,
                 const unsigned int dimZ = 1);

    void DISPATCHERDLL_API

    setInputData(float *dataPtr, const int numberOfBatches, const unsigned int dimX, const unsigned int dimY = 1,
                 const unsigned int dimZ = 1);

    void DISPATCHERDLL_API

    setInputData(double *dataPtr, const int numberOfBatches, const unsigned int dimX, const unsigned int dimY = 1,
                 const unsigned int dimZ = 1);

    void DISPATCHERDLL_API
    setInputData(float2
    * dataPtr,
    const int numberOfBatches,
    const unsigned int dimX,
    const unsigned int dimY = 1,
    const unsigned int dimZ = 1
    );

    void DISPATCHERDLL_API

    updateGraphNodeParameterAsync(const int nodeId, const std::string &parameterName, const bool parameterValue);

    void DISPATCHERDLL_API

    updateGraphNodeParameterAsync(const int nodeId, const std::string &parameterName, const int parameterValue);

    void DISPATCHERDLL_API

    updateGraphNodeParameterAsync(const int nodeId, const std::string &parameterName, const float parameterValue);

    void DISPATCHERDLL_API

    updateGraphNodeParameterAsync(const int nodeId, const std::string &parameterName,
                                  const std::string &parameterValue);

    std::vector <std::string> DISPATCHERDLL_API

    getAvailableDevicesNames();

    void DISPATCHERDLL_API

    setDevice(const std::string &deviceName);
};
