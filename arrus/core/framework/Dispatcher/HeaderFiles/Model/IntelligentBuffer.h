#pragma once

#include "DataPtr.h"
#include <vector>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>
#include <cuda_runtime.h>
#include "Model/GraphNodesLibrary/GraphNodes/CudaAssert.cuh"

enum ElementState {
    EMPTY,
    BUSY,
    FULL
};

struct IntelligentBufferElement {
    DataPtr ptr;
    ElementState state;
    std::function<void(DataPtr &ptr)> freeFunc;

    IntelligentBufferElement() : ptr(DataPtr()), state(ElementState::EMPTY),
                                 freeFunc([](DataPtr &ptr) {}) {};

    IntelligentBufferElement(DataPtr &ptr, ElementState state,
                             std::function<void(DataPtr &ptr)> freeFunc = [](DataPtr &ptr) {}) :
        ptr(ptr), state(state), freeFunc(freeFunc) {};
};

typedef unsigned int elementIndex;

class IntelligentBuffer {
private:
    std::vector <IntelligentBufferElement> dataBuffer;
    int currReadElementIndex, currWriteElementIndex;
    bool writerExitFlag;
    mutable boost::mutex mutex;
    boost::condition_variable readCondition, writeCondition;

public:
    IntelligentBuffer();

    ~IntelligentBuffer();

    void configure(int bufferSize);

    std::pair <elementIndex, DataPtr> getData();

    void setData(DataPtr &ptr, std::function<void(DataPtr &ptr)> freeFunc = [](DataPtr &ptr) {});

    void setDataEmpty(unsigned int index);

    void unlockReader();

    void unlockWriter();

    void unlockWriterToExit();
};

