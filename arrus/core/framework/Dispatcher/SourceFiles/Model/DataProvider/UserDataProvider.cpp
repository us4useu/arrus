#include "Model/DataProvider/UserDataProvider.h"

UserDataProvider::UserDataProvider(IntelligentBuffer *intelligentBuffer, DataPtr userInputData,
                                   const std::string &jsonConfig) : DataProvider() {
    this->userInputData = userInputData;
    this->intelligentBuffer = intelligentBuffer;
    this->intelligentBuffer->configure(this->userInputData.getPtrProperty("numberOfBatches").getValue<int>());
    this->iteration = -1;
    this->readTransmitInfo(jsonConfig);
    this->availableDataMutex.lock();
    this->runOnce = false;
}


UserDataProvider::~UserDataProvider() {
    if(this->userInputData.getDataSize() != 0)
        delete[] this->userInputData.getVoidPtr();
}

void UserDataProvider::preStart(const bool startOnce) {
    if(this->runOnce)
        this->availableDataMutex.unlock();
    this->runOnce = startOnce;
}

void UserDataProvider::preStop() {
    this->availableDataMutex.unlock();
}

int UserDataProvider::getBatchCount() {
    return this->userInputData.getDims().flatten();
}

void UserDataProvider::customProcess() {
    int numberOfBatches = this->userInputData.getPtrProperty("numberOfBatches").getValue<int>();
    this->iteration = (++this->iteration) % numberOfBatches;

    DataPtr batchPtr = this->userInputData;
    batchPtr.shiftByOffset(this->userInputData.getDims().flatten() * this->iteration);
    batchPtr.setDims(this->userInputData.getDims());
    this->addTransmitInfoToDataPtr(batchPtr, this->iteration % this->framesInfo.size());

    if(this->userInputData.getPtrProperty("iq").getValue<bool>()) batchPtr.setPtrProperty("iq", VariableAnyValue(true));

    this->intelligentBuffer->setData(batchPtr);

    if(this->runOnce)
        this->availableDataMutex.lock();
}

void UserDataProvider::SetUserData(DataPtr userInputData) {
    delete[] this->userInputData.getVoidPtr();

    this->userInputData = userInputData;
    this->iteration = -1;
}