#pragma once

#include "Model/DataProvider/DataProvider.h"
#include "Utils/CountingSemaphore.h"
//#include <Windows.h>
#include "HAL.h"

class HalDataProvider : public DataProvider, IHALCallback {
private:
    std::unique_ptr <IHAL, std::function<void(IHAL * )>> hal;

    void customProcess();

    void preStart(const bool startOnce);

    void preStop();

    void preKill();

    void checkHalDllVersion();

    CountingSemaphore dataAvailableCountingSemaphore;
    int currIdx;
    int lastSkippedChunks;

public:
    HalDataProvider(IntelligentBuffer *intelligentBuffer, const std::string &jsonConfig,
                    const std::string &halDeviceName);

    ~HalDataProvider();

    void OnNewData(int idx);

    int getBatchCount();
};

