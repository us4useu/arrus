#include "Model/GraphNodesLibrary/GraphNodes/InputDataGraphNode.h"
#include <cuda_runtime.h>
#include "Model/GraphNodesLibrary/GraphNodes/CudaAssert.cuh"

GraphNodesFactoryRegister<InputDataGraphNode> InputDataGraphNode::graphNodesFactoryRegister("inputData");

InputDataGraphNode::InputDataGraphNode()
{
	
}


InputDataGraphNode::~InputDataGraphNode()
{
	if (this->outputData.getDataSize() != 0)
	{
		CUDA_ASSERT(cudaFree(this->overlapData.getVoidPtr()));
		CUDA_ASSERT(cudaFree(this->outputData.getVoidPtr()));
		CUDA_ASSERT(cudaStreamDestroy(this->dataStream));
	}
}

void InputDataGraphNode::setIntelligentBuffer(IntelligentBuffer* intelligentBuffer)
{
	this->intelligentBuffer = intelligentBuffer;
}

void InputDataGraphNode::process(cudaStream_t& defaultStream)
{
	std::pair<elementIndex, DataPtr> srcData = this->intelligentBuffer->getData();

	if (srcData.second.getDataSize() != this->outputData.getDataSize())
	{
		CUDA_ASSERT(cudaStreamCreate(&this->dataStream));		
		this->overlapData = srcData.second;		
		CUDA_ASSERT(cudaMalloc(this->overlapData.getReferenceToVoidPtr(), srcData.second.getDataSize()));
		CUDA_ASSERT(cudaMemcpy(this->overlapData.getVoidPtr(), srcData.second.getVoidPtr(), srcData.second.getDataSize(), cudaMemcpyHostToDevice));
		this->intelligentBuffer->setDataEmpty(srcData.first);
		srcData = this->intelligentBuffer->getData();
		this->outputData = srcData.second;
		CUDA_ASSERT(cudaMalloc(this->outputData.getReferenceToVoidPtr(), srcData.second.getDataSize()));
	}

	std::swap(this->outputData, this->overlapData);
	std::swap(defaultStream, this->dataStream);

	this->overlapData.copyExtraData(srcData.second);
	this->overlapData.setDims(srcData.second.getDims());
	CUDA_ASSERT(cudaMemcpyAsync(this->overlapData.getVoidPtr(), srcData.second.getVoidPtr(), srcData.second.getDataSize(), cudaMemcpyHostToDevice, this->dataStream));
	InternalMemcpyCallbackData* callData = new InternalMemcpyCallbackData(this->intelligentBuffer, srcData.first);
	CUDA_ASSERT(cudaStreamAddCallback(this->dataStream, InputDataGraphNode::internalMemcpyCallback, callData, 0));
}

void CUDART_CB InputDataGraphNode::internalMemcpyCallback(cudaStream_t stream, cudaError_t status, void* callData)
{
	static_cast<InternalMemcpyCallbackData*>(callData)->process();
	delete callData;
}