#pragma once

#include "Model/GraphNodesLibrary/GraphNodes/GraphNode.h"
#include "Model/GraphNodesLibrary/GraphNodesFactory.h"
#include "Model/GraphNodesLibrary/GraphNodes/DataTranspositionGraphNode/CudaDataTranspositionGraphNode.cuh"

class DataTranspositionGraphNode : public GraphNode {
private:
    static GraphNodesFactoryRegister <DataTranspositionGraphNode> graphNodesFactoryRegister;

public:
    DataTranspositionGraphNode();

    ~DataTranspositionGraphNode();

    void process(cudaStream_t &defaultStream);

    template<typename T, typename K>
    void transformData(cudaStream_t defaultStream, T first, K second) {
        Dims inputDims = this->inputData.getDims();
        this->allocGPUMemory<K>(&this->outputData, Dims(inputDims.y, inputDims.x, inputDims.z));
        CudaDataTranspositionGraphNode::transposeData(this->inputData.getPtr<T *>(), this->outputData.getPtr<K *>(),
                                                      defaultStream,
                                                      inputDims.x, inputDims.y, inputDims.z);
    }
};

class InputDataTranspositionVisitator : public boost::static_visitor<void> {
private:
    DataTranspositionGraphNode *transposeNode;
    cudaStream_t defaultStream;
    bool rawDataFromHAL;

public:
    InputDataTranspositionVisitator(DataTranspositionGraphNode *transposeNode, cudaStream_t defaultStream,
                                    bool rawDataFromHAL) : transposeNode(transposeNode),
                                                           defaultStream(defaultStream),
                                                           rawDataFromHAL(rawDataFromHAL) {};

    template<typename T>
    void operator()(T &ptr) const {
        transposeNode->transformData(defaultStream, *ptr, *ptr);
    }

    void operator()(short *&ptr) const {
        if(rawDataFromHAL)
            transposeNode->transformData(defaultStream, (short) (0), (float) (0));
        else
            transposeNode->transformData(defaultStream, (short) (0), (short) (0));
    }

};

//template <>
//void InputDataTranspositionVisitator::operator()<short*>(short* ptr)
//{
//        if (rawDataFromHAL)
//                transposeNode->transformData(defaultStream, (short)(0), (float)(0));
//        else
//                transposeNode->transformData(defaultStream, (short)(0), (short)(0));
//}
