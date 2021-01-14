#pragma once

#include "Model/DataProvider/DataProvider.h"

class UserDataProvider : public DataProvider {
private:
    void customProcess();

    void preStart(const bool startOnce);

    void preStop();

    DataPtr userInputData;
    int iteration;
    bool runOnce;
    mutable boost::mutex availableDataMutex;
public:
    UserDataProvider(IntelligentBuffer *intelligentBuffer, DataPtr userInputData, const std::string &jsonConfig);

    ~UserDataProvider();

    void SetUserData(DataPtr userInputData);

    int getBatchCount();
};

