#include "Model/GraphNodesLibrary/GraphNodes/TgcGraphNode/CudaTgcGraphNode.cuh"
#include "Utils/DispatcherLogger.h"

class TgcFunction {
public:
    float getFunctionParameter(const std::unordered_map<std::string, float> &params, const std::string &paramName) {
        std::unordered_map<std::string, float>::const_iterator pos = params.find(paramName);
        if(pos == params.end()) {
            DISPATCHER_LOG(DispatcherLogType::ERROR_,
                           std::string("Missing ") + paramName + std::string(" parameter in tgc function."));
            return 0.0f;
        } else
            return pos->second;
    }
};

class LinearTgcFunction : TgcFunction {
private:
    float a, b;

public:
    LinearTgcFunction(const std::unordered_map<std::string, float> &params) {
        a = getFunctionParameter(params, "A");
        b = getFunctionParameter(params, "B");
    }

    __device__ float operator()(const float &arg) const {
        return a * arg + b;
    }
};

class ExponentialTgcFunction : TgcFunction {
private:
    float a, b, c;

public:
    ExponentialTgcFunction(const std::unordered_map<std::string, float> &params) {
        a = getFunctionParameter(params, "A");
        b = getFunctionParameter(params, "B");
        c = getFunctionParameter(params, "C");
    }

    __device__ float operator()(const float &arg) const {
        return a * expf(b * arg) + c;
    }
};

template<class UnaryFunction>
__global__ void
gpuTgc(const float *inputData, float *outputData, const int width, const int height, const float areaHeight,
       const UnaryFunction uf) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= (width * height))
        return;

    float currInputData = inputData[idx];
    int y = idx / width;
    float currDepth = (float) y / (float) (height - 1) * areaHeight;
    float currOutputData = currInputData * uf(currDepth);
    outputData[idx] = currOutputData;
}

void CudaTgcGraphNode::tgc(const float *inputData, float *output, const cudaStream_t &stream, const int width,
                           const int height, const float areaHeight,
                           const std::unordered_map<std::string, float> &params, const TGC_FUNC_TYPE tgcFuncType) {
    dim3 block(512);
    dim3 grid((width * height + block.x - 1) / block.x);

    if(tgcFuncType == TGC_FUNC_TYPE::LIN)
        gpuTgc << < grid, block, 0, stream >> >(inputData, output, width, height, areaHeight, LinearTgcFunction(
        params));
    else if(tgcFuncType == TGC_FUNC_TYPE::EXP)
        gpuTgc << < grid, block, 0, stream >> >(inputData, output, width, height, areaHeight, ExponentialTgcFunction(
        params));

    CUDA_ASSERT(cudaGetLastError());
}