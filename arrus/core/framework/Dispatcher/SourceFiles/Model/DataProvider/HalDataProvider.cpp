#include "Model/DataProvider/HalDataProvider.h"
#include "Utils/dllVersion.hpp"
#include "Utils/lastErrorAsString.hpp"
#include <algorithm>

HalDataProvider::HalDataProvider(IntelligentBuffer *intelligentBuffer, const std::string &jsonConfig,
                                 const std::string &halDeviceName) : DataProvider() {
    this->checkHalDllVersion();

    this->currIdx = 0;
    this->lastSkippedChunks = 0;
    this->hal = std::unique_ptr < IHAL, std::function < void(IHAL * )
        >> (GetHALInstance(halDeviceName.c_str()), std::mem_fn(&IHAL::Release));
    this->hal->RegisterCallback(this);
    this->hal->Configure(jsonConfig.c_str());
    this->readTransmitInfo(jsonConfig);
    this->intelligentBuffer = intelligentBuffer;
    this->intelligentBuffer->configure((int) this->framesInfo.size());
}

HalDataProvider::~HalDataProvider() {

}

void HalDataProvider::checkHalDllVersion() {
    WORD halDllVer[4];
    if(getDllVersion("USHAL.dll", halDllVer) == -1) {
        DispatcherLogger::getInstance()->log(DispatcherLogType::FATAL,
                                             std::string("Error during checking USHAL.dll version: ") +
                                             getLastErrorAsString());
        exit(-1);
    }

    WORD dispatcherDllVer[4];
    if(getDllVersion("Dispatcher.dll", dispatcherDllVer) == -1) {
        DispatcherLogger::getInstance()->log(DispatcherLogType::FATAL,
                                             std::string("Error during checking Dispatcher.dll version: ") +
                                             getLastErrorAsString());
        exit(-1);
    }

    if((halDllVer[0] != dispatcherDllVer[0]) || (halDllVer[1] != dispatcherDllVer[1])) {
        DispatcherLogger::getInstance()->log(DispatcherLogType::INFO,
                                             std::string("HAL version: ") + std::to_string(halDllVer[0]) + "." +
                                             std::to_string(halDllVer[1]) + "." +
                                             std::to_string(halDllVer[2]) + "." + std::to_string(halDllVer[3]));
        DispatcherLogger::getInstance()->log(DispatcherLogType::INFO,
                                             std::string("Dispatcher version: ") + std::to_string(dispatcherDllVer[0]) +
                                             "." + std::to_string(dispatcherDllVer[1]) + "." +
                                             std::to_string(dispatcherDllVer[2]) + "." +
                                             std::to_string(dispatcherDllVer[3]));
        DispatcherLogger::getInstance()->log(DispatcherLogType::FATAL,
                                             std::string("USHAL.dll and Dispatcher.dll versions are not compatible."));
        exit(-1);
    }
}

int HalDataProvider::getBatchCount() {
    int maxBatchCount = 0;
    for(int i = 0; i < this->framesInfo.size(); ++i)
        maxBatchCount = std::max(maxBatchCount, this->framesInfo[i]["samplesCount"].getValue<int>() *
                                                this->framesInfo[i]["eventsCount"].getValue<int>());
    return maxBatchCount * this->globalTransmitInfo["numReceivers"].getValue<int>();
}

void HalDataProvider::OnNewData(int idx) {
    this->dataAvailableCountingSemaphore.signal();
}

void HalDataProvider::customProcess() {
    this->dataAvailableCountingSemaphore.wait();

    if(!this->isWorking) {
        return;
    }
    short *data = (short *) this->hal->GetData(this->currIdx);
    int frameIdx = this->hal->GetMetadata(this->currIdx)->frameIdx;
    int skippedChunks = this->hal->GetMetadata(this->currIdx)->skippedChunks;

    DispatcherLogger::getInstance()->log(DispatcherLogType::INFO, std::string("FrameIdx: ") + std::to_string(frameIdx));

    if(this->lastSkippedChunks != skippedChunks) {
        this->lastSkippedChunks = skippedChunks;
        DispatcherLogger::getInstance()->log(DispatcherLogType::INFO,
                                             std::string("Dropped frames: ") + std::to_string(this->lastSkippedChunks));
    }

    DataPtr halDataPtr(data, Dims(this->globalTransmitInfo["numReceivers"].getValue<int>(),
                                  this->framesInfo[frameIdx]["samplesCount"].getValue<int>(),
                                  this->framesInfo[frameIdx]["eventsCount"].getValue<int>()));
    this->addTransmitInfoToDataPtr(halDataPtr, frameIdx);
    this->intelligentBuffer->setData(halDataPtr, [this, frameIdx](DataPtr &ptr) {
        hal->Sync(frameIdx);
        hal->SoftTrigger();
    });
    this->currIdx = (this->currIdx + 1) % this->framesInfo.size();
}

void HalDataProvider::preStart(const bool startOnce) {
    if(startOnce)
        this->hal->SoftTrigger();
    else
        this->hal->Start();
    this->hal->SoftTrigger();
}

void HalDataProvider::preStop() {
    this->hal->Stop();
    this->dataAvailableCountingSemaphore.signal();
}

void HalDataProvider::preKill() {
    this->hal->Release();
    this->dataAvailableCountingSemaphore.signal();
}
