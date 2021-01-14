#pragma once

#include "Model/GraphNodesLibrary/GraphNodesLibrary.h"
#include "Model/GraphThreadsLibrary/GraphThreadsLibrary.h"
#include "Model/IntelligentBuffer.h"
#include "Model/DataProvider/DataProvider.h"
#include "Model/DataProvider/UserDataProvider.h"
#include "Utils/DispatcherLogger.h"

class Model {
private:
    bool isRunning;
    std::string jsonConfig;
    std::string chosenHALDeviceName;

    std::vector <std::shared_ptr<GraphNodesLibrary>> graphNodesLibraries;
    std::unique_ptr <GraphThreadsLibrary> graphThreadsLibrary;
    IntelligentBuffer intelligentBuffer;
    std::unique_ptr <DataProvider> dataProvider;

    const int getCudaDeviceCount();

    void initializeEmptyGraph();

public:
    Model();

    ~Model();

    void setJsonConfig(const std::string &jsonConfig);

    void setChosenHALDeviceName(const std::string &halDeviceName);

    std::shared_ptr <GraphNodesLibrary> getGraphNodesLibrary(const int graphIndex);

    void start(const bool startOnce);

    void stop();

    void kill();

    void setCascadeMode();

    int getGraphNodesLibraryNumber();

    void synchronizeWithAllDevices();

    std::vector <std::string> getAvailableHALDevicesNames();

    bool activateUserDataProvider(DataPtr &userData);

    bool activateHalDataProvider();
};

