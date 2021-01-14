#include "Model/GraphNodesLibrary/GraphNodes/OutputDataGraphNode.h"
#include "Model/GraphNodesLibrary/GraphNodes/CudaAssert.cuh"
#include "Utils/DispatcherLogger.h"

GraphNodesFactoryRegister<OutputDataGraphNode> OutputDataGraphNode::graphNodesFactoryRegister("outputData");

OutputDataGraphNode::OutputDataGraphNode()
{	
	
}

OutputDataGraphNode::~OutputDataGraphNode()
{
	this->releaseMemory();	
}

void OutputDataGraphNode::releaseMemory()
{
	if (this->hostData.getDataSize() != 0)
	{
		CUDA_ASSERT( cudaFreeHost(this->hostData.getVoidPtr()) );
		CUDA_ASSERT( cudaFree(this->inputDataCopy.getVoidPtr()) );
		CUDA_ASSERT( cudaStreamDestroy(dataStream) );
	}
}

void OutputDataGraphNode::allocMemory()
{
	if (this->inputData.getDataSize() != this->hostData.getDataSize())
	{
		this->releaseMemory();
		CUDA_ASSERT(cudaStreamCreate(&dataStream));

		this->hostData = this->inputData;
		CUDA_ASSERT( cudaHostAlloc(this->hostData.getReferenceToVoidPtr(), this->inputData.getDataSize(), cudaHostAllocPortable) );
		this->inputDataCopy = this->inputData;
		CUDA_ASSERT( cudaMalloc(this->inputDataCopy.getReferenceToVoidPtr(), this->inputData.getDataSize()) );
	}
}

void OutputDataGraphNode::process(cudaStream_t& defaultStream)
{
	this->allocMemory();

	this->inputDataCopy.copyExtraData(this->inputData);
	this->inputDataCopy.setDims(this->inputData.getDims());
	CUDA_ASSERT( cudaMemcpyAsync(this->inputDataCopy.getVoidPtr(), this->inputData.getVoidPtr(), this->inputData.getDataSize(), cudaMemcpyDeviceToDevice, defaultStream) );
	CUDA_ASSERT( cudaStreamSynchronize(defaultStream) );
	CUDA_ASSERT( cudaStreamSynchronize(this->dataStream) );
	this->hostData.copyExtraData(this->inputDataCopy);
	this->hostData.setDims(this->inputDataCopy.getDims());
	CUDA_ASSERT( cudaMemcpyAsync(this->hostData.getVoidPtr(), this->inputDataCopy.getVoidPtr(), this->inputDataCopy.getDataSize(), cudaMemcpyDeviceToHost, this->dataStream) );
	CUDA_ASSERT( cudaStreamAddCallback(this->dataStream, OutputDataGraphNode::callback, this, 0) );

	this->outputData = this->inputData;
}

void OutputDataGraphNode::registerCallback(const boost::function<void(void* data, int iterationId, int graphNodeId, int dimX, int dimY, int dimZ, int dataType)> callbackFunc)
{
	this->callbackFunc = callbackFunc;
}

void CUDART_CB OutputDataGraphNode::callback(cudaStream_t stream, cudaError_t status, void* userData)
{
	OutputDataGraphNode* graphNode = static_cast<OutputDataGraphNode*>(userData);
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