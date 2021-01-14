#include "Model/GraphNodesLibrary/GraphNodes/ExternTransformGraphNode.h"
#include "Model/GraphNodesLibrary/GraphNodes/CudaAssert.cuh"
#include "Utils/DispatcherLogger.h"

GraphNodesFactoryRegister<ExternTransformGraphNode> ExternTransformGraphNode::graphNodesFactoryRegister("externTransform");

ExternTransformGraphNode::ExternTransformGraphNode()
{	
	
}

ExternTransformGraphNode::~ExternTransformGraphNode()
{
	this->releaseMemory();	
}

void ExternTransformGraphNode::releaseMemory()
{
	if (this->hostData.getDataSize() != 0)
	{
		CUDA_ASSERT( cudaFreeHost(this->hostData.getVoidPtr()) );
		CUDA_ASSERT( cudaFree(this->outputData.getVoidPtr()) );
	}
}

void ExternTransformGraphNode::allocMemory()
{
	if (this->inputData.getDataSize() != this->hostData.getDataSize())
	{
		this->releaseMemory();

		this->hostData = this->inputData;
		CUDA_ASSERT( cudaHostAlloc(this->hostData.getReferenceToVoidPtr(), this->inputData.getDataSize(), cudaHostAllocPortable) );
		this->outputData = this->inputData;
		CUDA_ASSERT( cudaMalloc(this->outputData.getReferenceToVoidPtr(), this->inputData.getDataSize()) );
	}
}

void ExternTransformGraphNode::process(cudaStream_t& defaultStream)
{
	this->allocMemory();

	this->hostData.copyExtraData(this->inputData);
	this->hostData.setDims(this->inputData.getDims());
	CUDA_ASSERT( cudaMemcpyAsync(this->hostData.getVoidPtr(), this->inputData.getVoidPtr(), this->inputData.getDataSize(), cudaMemcpyDeviceToHost, defaultStream) );
	CUDA_ASSERT( cudaStreamAddCallback(defaultStream, ExternTransformGraphNode::callback, this, 0) );
	CUDA_ASSERT( cudaMemcpyAsync(this->outputData.getVoidPtr(), this->hostData.getVoidPtr(), this->hostData.getDataSize(), cudaMemcpyHostToDevice, defaultStream) );
	this->outputData.copyExtraData(this->inputData);
	this->outputData.setDims(this->inputData.getDims());
}

void ExternTransformGraphNode::registerCallback(const boost::function<void(void* data, int iterationId, int graphNodeId, int dimX, int dimY, int dimZ, int dataType)> callbackFunc)
{
	this->callbackFunc = callbackFunc;
}

void CUDART_CB ExternTransformGraphNode::callback(cudaStream_t stream, cudaError_t status, void* userData)
{
	ExternTransformGraphNode* graphNode = static_cast<ExternTransformGraphNode*>(userData);
	DataPtr hostData = graphNode->hostData;
	try
	{
		graphNode->callbackFunc(hostData.getVoidPtr(), hostData.getPtrProperty("dataId").getValue<int>(), graphNode->getNodeId(), hostData.getDims().x, 
			hostData.getDims().y, hostData.getDims().z, hostData.getRawPtr().which());
	}
	catch (boost::bad_function_call exception)
	{
		DISPATCHER_LOG(DispatcherLogType::WARNING, std::string("Problem with callback occured in node with id: ") + std::to_string(graphNode->getNodeId()) + std::string(". No data returned."));
	}
}