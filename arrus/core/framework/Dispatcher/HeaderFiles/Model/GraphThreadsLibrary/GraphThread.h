#pragma once

#include <boost/thread.hpp>
#include "Model/GraphNodesLibrary/GraphNodes/GraphNode.h"
#include "Model/GraphNodesLibrary/GraphNodes/CudaAssert.cuh"

// forward declaration of class GraphThreadsLibrary to avoid circular dependency
class GraphThreadsLibrary;

class GraphThread {
private:
    GraphThreadsLibrary *graphThreadsLibrary;
    std::shared_ptr <GraphNode> startingNode;
    boost::thread myThread;
    mutable boost::mutex myMutex;
    cudaStream_t defaultStream;
    int myThreadId;
    bool isWorking;
    bool isContinousWork;

    void process();

    void iterateOverGraph();

public:
    GraphThread(const int id, GraphThreadsLibrary *graphThreadsLibrary);

    ~GraphThread();

    void startFromNode(const std::shared_ptr <GraphNode> node);

    void continousWorkFromNode(const std::shared_ptr <GraphNode> node);

    void start();

    void join();

    void kill();
};

