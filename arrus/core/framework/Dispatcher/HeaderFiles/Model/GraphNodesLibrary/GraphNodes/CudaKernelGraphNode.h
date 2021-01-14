#pragma once

#include "Model/GraphNodesLibrary/GraphNodes/GraphNode.h"
#include "Model/GraphNodesLibrary/GraphNodesFactory.h"

#include <nvrtc.h>
#include <cuda.h>

class CudaKernelGraphNode : public GraphNode {
private:
    static GraphNodesFactoryRegister <CudaKernelGraphNode> graphNodesFactoryRegister;

    std::string loadCudaKernelFromFile(const std::string &kernelFilePath);

    void checkCudaKernelCompilationLog(const nvrtcProgram &prog);

    std::vector<char> loadPTX(const nvrtcProgram &prog);

    void compileUserCudaKernel();

    void loadUserGlobalFunction(const std::string &cudaKernel);

    std::vector<void *> getPointersToCudaKernelArguments(std::vector <VariableAnyValue> &inputArgs);

    CUdeviceptr getInputDataPtr();

    void allocOutputData();

    CUmodule module;
    CUfunction kernel;
    bool isUserKernelCompiled;

public:
    CudaKernelGraphNode();

    ~CudaKernelGraphNode();

    void process(cudaStream_t &defaultStream);
};

