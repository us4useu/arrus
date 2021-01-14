#pragma once

#include "Model/GraphNodesLibrary/GraphNodes/GraphNode.h"
#include "Model/GraphNodesLibrary/GraphNodesFactory.h"
#include "Model/GraphNodesLibrary/GraphNodes/Filter1DGraphNode/CudaFilter1DGraphNode.cuh"

class Filter1DGraphNode : public GraphNode {
private:
    static GraphNodesFactoryRegister <Filter1DGraphNode> graphNodesFactoryRegister;
    CudaFilter1DGraphNode cudaFilter1DGraphNode;
    std::vector<float> feedforwardCoefficients, feedbackCoefficients, feedbackCoefficientsMatrixAsVector;
    std::string feedforwardCoefficientsFileName, feedbackCoefficientsFileName;

    void readFilterCoefficientsFromFile(const std::string &pathToFile, std::vector<float> &filterCoefficients);

    void readFilterCoefficientsFromTxtFile(const std::string &pathToFile, std::vector<float> &filterCoefficients);

    void readFilterCoefficientsFromBinaryFile(const std::string &pathToFile, std::vector<float> &filterCoefficients);

    void loadFilterCoefficients(cudaStream_t &defaultStream);

    std::vector<float> getNormalizedCoefficients(const std::vector<float> &inputCoeff, const float normalizeFactor);

    void buildFeedbackCoefficientsMatrix(const std::vector<float> &feedbackCoefficients, cudaStream_t &defaultStream);

public:
    Filter1DGraphNode();

    ~Filter1DGraphNode();

    void process(cudaStream_t &defaultStream);
};

