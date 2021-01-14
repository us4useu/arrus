#include "Dispatcher.h"

#include "Controller/ControllerEvents/BuildEvent.h"
#include "Controller/ControllerEvents/BindCallbackEvent.h"
#include "Controller/ControllerEvents/StartModelEvent.h"
#include "Controller/ControllerEvents/StopModelEvent.h"
#include "Controller/ControllerEvents/GetAvailableDevicesNamesEvent.h"
#include "Controller/ControllerEvents/SetDeviceEvent.h"
#include <cuda_runtime.h>

Dispatcher::Dispatcher() {
    this->controller = new Controller();
    this->controller->start();
}

Dispatcher::~Dispatcher() {
    this->controller->stop();
    delete this->controller;
    cudaDeviceReset();
}

void Dispatcher::startAsync() {
    this->controller->sendControllerEvent(std::shared_ptr<StartModelEvent>(new StartModelEvent()));
}

void Dispatcher::startOnce() {
    this->controller->sendControllerEvent(std::shared_ptr<StartModelEvent>(new StartModelEvent(true)));
    this->controller->synchronize();
}

void Dispatcher::stop() {
    this->controller->sendControllerEvent(std::shared_ptr<StopModelEvent>(new StopModelEvent()));
    this->controller->synchronize();
}

void Dispatcher::kill() {
    this->controller->kill();
}

void Dispatcher::registerCallback(const int id,
                                  void(*callback)(void *data, int iterationId, int graphNodeId, int dimX, int dimY,
                                                  int dimZ, int dataType)) {
    this->controller->sendControllerEvent(std::shared_ptr<BindCallbackEvent>(new BindCallbackEvent(id, callback)));
    this->controller->synchronize();
}

bool Dispatcher::build(const std::string &json, const bool cascadeMode) {
    std::shared_ptr <BuildEvent> inputEvent(new BuildEvent(json, cascadeMode));
    this->controller->sendControllerEvent(inputEvent);
    this->controller->synchronize();
    return inputEvent->getSucceeded();
}

void Dispatcher::buildAsync(const std::string &json, const bool cascadeMode) {
    this->controller->sendControllerEvent(std::shared_ptr<BuildEvent>(new BuildEvent(json, cascadeMode)));
}

void
Dispatcher::setInputData(short *dataPtr, const int numberOfBatches, const unsigned int dimX, const unsigned int dimY,
                         const unsigned int dimZ) {
    this->internalSetInputData(dataPtr, numberOfBatches, dimX, dimY, dimZ);
}

void Dispatcher::setInputData(int *dataPtr, const int numberOfBatches, const unsigned int dimX, const unsigned int dimY,
                              const unsigned int dimZ) {
    this->internalSetInputData(dataPtr, numberOfBatches, dimX, dimY, dimZ);
}

void
Dispatcher::setInputData(float *dataPtr, const int numberOfBatches, const unsigned int dimX, const unsigned int dimY,
                         const unsigned int dimZ) {
    this->internalSetInputData(dataPtr, numberOfBatches, dimX, dimY, dimZ);
}

void
Dispatcher::setInputData(double *dataPtr, const int numberOfBatches, const unsigned int dimX, const unsigned int dimY,
                         const unsigned int dimZ) {
    this->internalSetInputData(dataPtr, numberOfBatches, dimX, dimY, dimZ);
}

void
Dispatcher::setInputData(float2 *dataPtr, const int numberOfBatches, const unsigned int dimX, const unsigned int dimY,
                         const unsigned int dimZ) {
    this->internalSetInputData(dataPtr, numberOfBatches, dimX, dimY, dimZ);
}

void Dispatcher::updateGraphNodeParameterAsync(const int nodeId, const std::string &parameterName,
                                               const bool parameterValue) {
    this->internalUpdateGraphNodeParameterAsync(nodeId, parameterName, parameterValue);
}

void Dispatcher::updateGraphNodeParameterAsync(const int nodeId, const std::string &parameterName,
                                               const int parameterValue) {
    this->internalUpdateGraphNodeParameterAsync(nodeId, parameterName, parameterValue);
}

void Dispatcher::updateGraphNodeParameterAsync(const int nodeId, const std::string &parameterName,
                                               const float parameterValue) {
    this->internalUpdateGraphNodeParameterAsync(nodeId, parameterName, parameterValue);
}

void Dispatcher::updateGraphNodeParameterAsync(const int nodeId, const std::string &parameterName,
                                               const std::string &parameterValue) {
    this->internalUpdateGraphNodeParameterAsync(nodeId, parameterName, parameterValue);
}

std::vector <std::string> Dispatcher::getAvailableDevicesNames() {
    std::shared_ptr <GetAvailableDevicesNamesEvent> inputEvent(new GetAvailableDevicesNamesEvent());
    this->controller->sendControllerEvent(inputEvent);
    this->controller->synchronize();
    return inputEvent->getDevicesNames();
}

void Dispatcher::setDevice(const std::string &deviceName) {
    this->controller->sendControllerEvent(std::shared_ptr<SetDeviceEvent>(new SetDeviceEvent(deviceName)));
}