#include "Model/Model.h"
#include "Model/GraphNodesLibrary/GraphNodesFactory.h"
#include "Model/GraphNodesLibrary/GraphNodes/InputDataGraphNode.h"
#include "Model/DataProvider/HalDataProvider.h"

Model::Model() {
    this->graphThreadsLibrary = std::unique_ptr<GraphThreadsLibrary>(
        new GraphThreadsLibrary(this->getCudaDeviceCount()));
    this->initializeEmptyGraph();
    this->isRunning = false;
}

Model::~Model() {
}

void Model::initializeEmptyGraph() {
    this->graphNodesLibraries.push_back(std::shared_ptr<GraphNodesLibrary>(new GraphNodesLibrary()));
    std::shared_ptr <GraphNode> graphNode = GraphNodesFactory::createGraphNode("inputData");
    this->graphNodesLibraries.back()->registerNode(STARTING_NODE_ID, graphNode);
    std::dynamic_pointer_cast<InputDataGraphNode>(graphNode)->setIntelligentBuffer(&this->intelligentBuffer);
}

bool Model::activateUserDataProvider(DataPtr &userData) {
    if(jsonConfig.empty()) {
        DISPATCHER_LOG(DispatcherLogType::ERROR_, "Build operation must be called first.");
        return false;
    }
    if(this->dataProvider == nullptr) {
        this->dataProvider = std::unique_ptr<DataProvider>(
            new UserDataProvider(&this->intelligentBuffer, userData, this->jsonConfig));
    } else {
        UserDataProvider *ptr = dynamic_cast<UserDataProvider *>(this->dataProvider.get());
        ptr->SetUserData(userData);
    }
    return true;
}

bool Model::activateHalDataProvider() {
    if(this->jsonConfig.empty()) {
        DISPATCHER_LOG(DispatcherLogType::ERROR_, "Build operation must be called first.");
        return false;
    }
    if(this->chosenHALDeviceName.empty()) {
        std::vector <std::string> halNames = this->getAvailableHALDevicesNames();
        if(halNames.empty()) {
            DISPATCHER_LOG(DispatcherLogType::ERROR_, "There aren't any available devices.");
            return false;
        }
        this->chosenHALDeviceName = halNames[0];
        if(halNames.size() > 1) {
            DISPATCHER_LOG(DispatcherLogType::INFO,
                           "There are " + std::to_string(halNames.size()) + " available devices. The Dispatcher" +
                           " is launched on the " + this->chosenHALDeviceName + " device.");
        }
    }
    this->dataProvider = std::unique_ptr<DataProvider>(
        new HalDataProvider(&this->intelligentBuffer, this->jsonConfig, this->chosenHALDeviceName));
    return true;
}

void Model::setJsonConfig(const std::string &jsonConfig) {
    this->jsonConfig = jsonConfig;
}

void Model::setChosenHALDeviceName(const std::string &halDeviceName) {
    this->chosenHALDeviceName = halDeviceName;
}

const int Model::getCudaDeviceCount() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

std::vector <std::string> Model::getAvailableHALDevicesNames() {
    char **listOfNames;
    int namesNumber = GetAvailableHALDevicesNames(&listOfNames);

    std::vector <std::string> names(namesNumber);
    for(int i = 0; i < namesNumber; ++i)
        names[i] = std::string(listOfNames[i]);

    for(int i = 0; i < namesNumber; ++i)
        delete[] listOfNames[i];
    delete[] listOfNames;

    return names;
}

std::shared_ptr <GraphNodesLibrary> Model::getGraphNodesLibrary(const int graphIndex) {
    return this->graphNodesLibraries[graphIndex];
}

void Model::start(const bool startOnce) {
    if(this->dataProvider == nullptr)
        if(!this->activateHalDataProvider())
            return;

    if(!this->isRunning) {
        this->graphThreadsLibrary->setGraphNodesLibraries(this->graphNodesLibraries);
        this->graphThreadsLibrary->start();
        this->isRunning = true;
    }
    this->dataProvider->start(startOnce);
}

void Model::stop() {
    this->graphThreadsLibrary->stop();
    this->synchronizeWithAllDevices();
    if(this->dataProvider)
        this->dataProvider->stop();
    this->isRunning = false;
}

void Model::synchronizeWithAllDevices() {
    int deviceCount = this->getCudaDeviceCount();
    for(int i = 0; i < deviceCount; ++i) {
        CUDA_ASSERT(cudaSetDevice(i));
        CUDA_ASSERT(cudaDeviceSynchronize());
    }
}

void Model::setCascadeMode() {
    int deviceCount = this->getCudaDeviceCount();
    for(int i = 1; i < deviceCount; ++i)
        this->initializeEmptyGraph();
}

int Model::getGraphNodesLibraryNumber() {
    return (int) this->graphNodesLibraries.size();
}

void Model::kill() {
    this->graphThreadsLibrary->kill();
    this->dataProvider->kill();
}
