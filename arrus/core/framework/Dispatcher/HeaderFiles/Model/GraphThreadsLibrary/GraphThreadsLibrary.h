#pragma once

#include "Model/GraphNodesLibrary/GraphNodesLibrary.h"
#include <boost/thread/recursive_mutex.hpp>
#include "Model/GraphThreadsLibrary/GraphThread.h"

typedef int threadId;

class GraphThreadsLibrary {
private:
    std::vector <std::shared_ptr<GraphNodesLibrary>> graphNodesLibraries;
    std::vector <std::unique_ptr<GraphThread>> graphThreads;
    int activeThreadsNumber;
    mutable boost::recursive_mutex activeThreadsMutex;

    void startCascade();

    void plainStart();

public:
    GraphThreadsLibrary();

    GraphThreadsLibrary(const int cudaDeviceCount);

    ~GraphThreadsLibrary();

    void start();

    void startThread(const std::shared_ptr <GraphNode> startingNode, const int threadId);

    void suspendThread();

    void stop();

    void kill();

    void setGraphNodesLibraries(const std::vector <std::shared_ptr<GraphNodesLibrary>> &graphNodesLibraries);

    void updateGraphNodesLibrary(const int id);
};

