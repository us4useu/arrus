#include "Model/GraphNodesLibrary/GraphNodes/Filter1DGraphNode/CudaFilter1DGraphNode.cuh"
#include "Model/GraphNodesLibrary/CudaUtils/CudaTranspositionUtils.cuh"
#include "Model/GraphNodesLibrary/CudaUtils/CudaVectorMathUtils.cuh"
#include "Utils/DispatcherLogger.h"

__device__ __constant__ float deviceFeedforwardCoefficients[2048];
__device__ __constant__ float deviceFeedbackCoefficients[2048];
__device__ __constant__ float deviceFeedbackCoefficientsMatrix[2048];

template <typename T>
__global__ void gpuFir(const T *input, T *output, const int feedforwardFilterSize, const int extendedFeedforwardFilterSize, const int batchLength, const int maxThreadId)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int ch = idx / batchLength;
	int n = idx % batchLength;

	extern __shared__ float extCachedInputData[];
	T* cachedInputData = (T*)extCachedInputData;

	for (int i = n - extendedFeedforwardFilterSize, localIdx = threadIdx.x; localIdx < (extendedFeedforwardFilterSize + blockDim.x); i += blockDim.x, localIdx += blockDim.x)
	{
		if (i < 0)
			cachedInputData[localIdx] = makeZeroValue<T>();
		else
			cachedInputData[localIdx] = input[ch * batchLength + i];
	}

	__syncthreads();

	if (idx >= maxThreadId)
		return;

	T result = makeZeroValue<T>();
	int localN = threadIdx.x + extendedFeedforwardFilterSize;
	for (int i = 0; i < feedforwardFilterSize; ++i)
	{
		result += cachedInputData[localN - i] * deviceFeedforwardCoefficients[i];
	}

	output[idx] = result;
}

// buffering in shared memory is efficient for small sizes of filter coefficients, but kills ocupancy for larger filters
template <typename T>
__global__ void gpuZeroPrologueIIR(const T *input, T *output, const int batchCount, const int feedbackFilterSize, const int maxThreadId)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= maxThreadId)
		return;

	int dataBlockId = idx / batchCount;
	int idInDataBlock = idx % batchCount;

	int outputBufferOffset = feedbackFilterSize - (IIR_DATA_BLOCK_COUNT % feedbackFilterSize);
	T* dataBlockOutput = output + idInDataBlock + batchCount * dataBlockId * feedbackFilterSize;
	const T* dataBlockInput = input + idInDataBlock;

	for (int i = dataBlockId * IIR_DATA_BLOCK_COUNT, it = 0; it < IIR_DATA_BLOCK_COUNT; ++i, ++it)
	{
		T prefetchedInput = dataBlockInput[batchCount * i];
		T accumulator = makeZeroValue<T>();
		int moduloCorrection = 0;
		for (int currFilterSize = feedbackFilterSize; currFilterSize > 0; --currFilterSize)
		{
			if (it + outputBufferOffset - currFilterSize - moduloCorrection >= feedbackFilterSize)
				moduloCorrection = (it + outputBufferOffset - currFilterSize) / feedbackFilterSize * feedbackFilterSize;
			if (it >= currFilterSize)
			{
				int prevOutputDataIndex = it + outputBufferOffset - currFilterSize - moduloCorrection; // (it - currFilterSize) % feedbackFilterSize
				accumulator -= deviceFeedbackCoefficients[currFilterSize] * dataBlockOutput[batchCount * prevOutputDataIndex];
			}
		}
		int currCachedOutput = it + outputBufferOffset - moduloCorrection;
		currCachedOutput = (currCachedOutput == feedbackFilterSize) ? 0 : currCachedOutput;

		dataBlockOutput[batchCount * currCachedOutput] = prefetchedInput + accumulator;
	}
}

template <typename T>
__global__ void gpuPrologueTransfer(T *prologues, const int maxThreadId, const int feedbackFilterSize, const int dataBlocksInBatch, const int batchCount)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= maxThreadId)
		return;

	extern __shared__ float extCache[];
	T* cache = (T*)extCache;

	for (int it = 0; it < dataBlocksInBatch - 1; ++it)
	{
		cache[threadIdx.x + blockDim.x * threadIdx.y] = prologues[idx + (it * feedbackFilterSize + threadIdx.y) * batchCount];

		__syncthreads();

		T filteredSample = makeZeroValue<T>();
		for (int j = 0; j < feedbackFilterSize; ++j)
			filteredSample += deviceFeedbackCoefficientsMatrix[j * feedbackFilterSize + threadIdx.y] * cache[threadIdx.x + blockDim.x * j];

		prologues[idx + ((it + 1) * feedbackFilterSize + threadIdx.y) * batchCount] += filteredSample;

		__syncthreads();
	}
}

template <typename T>
__global__ void gpuWithPrologueIIR(const T *input, T *output, const T *prologues, const int batchCount, const int batchLength, const int feedbackFilterSize, const int maxThreadId)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= maxThreadId)
		return;

	int dataBlockId = idx / batchCount;
	int idInDataBlock = idx % batchCount;

	int endOfDataBlock = (dataBlockId + 1) * IIR_DATA_BLOCK_COUNT;
	endOfDataBlock = (endOfDataBlock > batchLength) ? batchLength : endOfDataBlock;

	for (int i = dataBlockId * IIR_DATA_BLOCK_COUNT, it = 0; i < endOfDataBlock; ++i, ++it)
	{
		T prefetchedInput = input[idInDataBlock + batchCount * i];
		T accumulator = makeZeroValue<T>();
		for (int currFilterSize = feedbackFilterSize; currFilterSize > 0; --currFilterSize)
		{
			if (it >= currFilterSize)
			{
				accumulator -= deviceFeedbackCoefficients[currFilterSize] * output[idInDataBlock + batchCount * (i - currFilterSize)];
			}
			else if (dataBlockId != 0)
			{
				accumulator -= deviceFeedbackCoefficients[currFilterSize] *
					prologues[((dataBlockId - 1) * feedbackFilterSize + (feedbackFilterSize - currFilterSize) + it) * batchCount + idInDataBlock];
			}
		}
		output[idInDataBlock + batchCount * i] = accumulator + prefetchedInput;
	}
}

CudaFilter1DGraphNode::CudaFilter1DGraphNode()
{
	this->prologuesBuffer = nullptr;
	this->prologuesIqBuffer = nullptr;
	this->firOutputBuffer = nullptr;
	this->firIqOutputBuffer = nullptr;
	this->prologuesBufferSize = 0;
	this->prologuesIqBufferSize = 0;
	this->firOutputBufferSize = 0;
	this->firIqOutputBufferSize = 0;
}

__host__ CudaFilter1DGraphNode::~CudaFilter1DGraphNode()
{
	this->releaseMemory(this->firOutputBuffer);
	this->releaseMemory(this->prologuesBuffer);
	this->releaseMemory(this->firIqOutputBuffer);
	this->releaseMemory(this->prologuesIqBuffer);
}

void CudaFilter1DGraphNode::fir(const float *inputPtr, float *outputPtr, const cudaStream_t& stream, const int feedforwardFilterSize,
								const int dataCount, const int batchLength)
{
	dim3 filterBlockDim(512);
	dim3 filterGridDim((dataCount + filterBlockDim.x - 1) / filterBlockDim.x);
	int extendedFeedforwardFilterSize = ((feedforwardFilterSize + 31) / 32) * 32;
	int externalSharedMemorySize = (filterBlockDim.x + extendedFeedforwardFilterSize) * sizeof(float);
	gpuFir << <filterGridDim, filterBlockDim, externalSharedMemorySize, stream >> >(inputPtr, outputPtr, feedforwardFilterSize, extendedFeedforwardFilterSize, batchLength, dataCount);
	CUDA_ASSERT(cudaGetLastError());
}

void CudaFilter1DGraphNode::firIq(const float2 *inputPtr, float2 *outputPtr, const cudaStream_t& stream, const int feedforwardFilterSize,
								  const int dataCount, const int batchLength)
{
	dim3 filterBlockDim(512);
	dim3 filterGridDim((dataCount + filterBlockDim.x - 1) / filterBlockDim.x);
	int extendedFeedforwardFilterSize = ((feedforwardFilterSize + 15) / 16) * 16;
	int externalSharedMemorySize = (filterBlockDim.x + extendedFeedforwardFilterSize) * sizeof(float2);
	gpuFir << <filterGridDim, filterBlockDim, externalSharedMemorySize, stream >> >(inputPtr, outputPtr, feedforwardFilterSize, extendedFeedforwardFilterSize, batchLength, dataCount);
	CUDA_ASSERT(cudaGetLastError());
}

void CudaFilter1DGraphNode::iir(const float *inputPtr, float *outputPtr, const cudaStream_t& stream, const int feedforwardFilterSize,
								const int feedbackFilterSize, const int dataCount, const int batchLength)
{
	int currFirOutputBufferSize = dataCount * sizeof(float);
	if (currFirOutputBufferSize > this->firOutputBufferSize)
	{
		this->firOutputBufferSize = currFirOutputBufferSize;
		this->allocMemory(currFirOutputBufferSize, &this->firOutputBuffer);
	}

	this->fir(inputPtr, this->firOutputBuffer, stream, feedforwardFilterSize, dataCount, batchLength);

	int batchCount = dataCount / batchLength;
	CudaTranspositionUtils::transpose(this->firOutputBuffer, outputPtr, batchLength, batchCount, stream);

	dim3 block(256);
	int dataBlocksInBatch = (((batchLength + IIR_DATA_BLOCK_COUNT - 1) / IIR_DATA_BLOCK_COUNT) - 1);
	int allThreads = dataBlocksInBatch * batchCount;

	int currProloguesBufferSize = allThreads * sizeof(float) * feedbackFilterSize;
	if (currProloguesBufferSize > this->prologuesBufferSize)
	{
		this->prologuesBufferSize = currProloguesBufferSize;
		this->allocMemory(currProloguesBufferSize, &this->prologuesBuffer);
	}

	dim3 grid((allThreads + block.x - 1) / block.x);
	gpuZeroPrologueIIR << <grid, block, 0, stream >> >(outputPtr, this->prologuesBuffer, batchCount, feedbackFilterSize, allThreads);
	CUDA_ASSERT(cudaGetLastError());

	if (feedbackFilterSize > 32)
	{
		DISPATCHER_LOG(DispatcherLogType::ERROR_, std::string("Maximum number of feedback coefficients is 33."));
		return;
	}

	dim3 prologueBlock(32, feedbackFilterSize);
	dim3 prologueGrid((batchCount + prologueBlock.x - 1) / prologueBlock.x, 1);
	int externalSharedMemorySize = prologueBlock.x * prologueBlock.y * sizeof(float);
	gpuPrologueTransfer << <prologueGrid, prologueBlock, externalSharedMemorySize, stream >> >(this->prologuesBuffer, batchCount, feedbackFilterSize, dataBlocksInBatch, batchCount);
	CUDA_ASSERT(cudaGetLastError());

	allThreads = (dataBlocksInBatch + 1) * batchCount;
	grid = dim3((allThreads + block.x - 1) / block.x);
	gpuWithPrologueIIR << <grid, block, 0, stream >> >(outputPtr, this->firOutputBuffer, this->prologuesBuffer, batchCount, batchLength, feedbackFilterSize, allThreads);
	CUDA_ASSERT(cudaGetLastError());

	CudaTranspositionUtils::transpose(this->firOutputBuffer, outputPtr, batchCount, batchLength, stream);
}

void CudaFilter1DGraphNode::iirIq(const float2 *inputPtr, float2 *outputPtr, const cudaStream_t& stream, const int feedforwardFilterSize,
								  const int feedbackFilterSize, const int dataCount, const int batchLength)
{
	int currFirIqOutputBufferSize = dataCount * sizeof(float2);
	if (currFirIqOutputBufferSize > this->firIqOutputBufferSize)
	{
		this->firIqOutputBufferSize = currFirIqOutputBufferSize;
		this->allocMemory(currFirIqOutputBufferSize, &this->firIqOutputBuffer);
	}

	this->firIq(inputPtr, this->firIqOutputBuffer, stream, feedforwardFilterSize, dataCount, batchLength);

	int batchCount = dataCount / batchLength;
	CudaTranspositionUtils::transpose(this->firIqOutputBuffer, outputPtr, batchLength, batchCount, stream);

	dim3 block(256);
	int dataBlocksInBatch = (((batchLength + IIR_DATA_BLOCK_COUNT - 1) / IIR_DATA_BLOCK_COUNT) - 1);
	int allThreads = dataBlocksInBatch * batchCount;

	int currProloguesIqBufferSize = allThreads * sizeof(float2) * feedbackFilterSize;
	if (currProloguesIqBufferSize > this->prologuesIqBufferSize)
	{
		this->prologuesIqBufferSize = currProloguesIqBufferSize;
		this->allocMemory(currProloguesIqBufferSize, &this->prologuesIqBuffer);
	}

	dim3 grid((allThreads + block.x - 1) / block.x);
	gpuZeroPrologueIIR << <grid, block, 0, stream >> >(outputPtr, this->prologuesIqBuffer, batchCount, feedbackFilterSize, allThreads);
	CUDA_ASSERT(cudaGetLastError());

	if (feedbackFilterSize > 32)
	{
		DISPATCHER_LOG(DispatcherLogType::ERROR_, std::string("Maximum number of feedback coefficients is 33."));
		return;
	}

	dim3 prologueBlock(32, feedbackFilterSize);
	dim3 prologueGrid((batchCount + prologueBlock.x - 1) / prologueBlock.x, 1);
	int externalSharedMemorySize = prologueBlock.x * prologueBlock.y * sizeof(float2);
	gpuPrologueTransfer << <prologueGrid, prologueBlock, externalSharedMemorySize, stream >> >(this->prologuesIqBuffer, batchCount, feedbackFilterSize, dataBlocksInBatch, batchCount);
	CUDA_ASSERT(cudaGetLastError());

	allThreads = (dataBlocksInBatch + 1) * batchCount;
	grid = dim3((allThreads + block.x - 1) / block.x);
	gpuWithPrologueIIR << <grid, block, 0, stream >> >(outputPtr, this->firIqOutputBuffer, this->prologuesIqBuffer, batchCount, batchLength, feedbackFilterSize, allThreads);
	CUDA_ASSERT(cudaGetLastError());

	CudaTranspositionUtils::transpose(this->firIqOutputBuffer, outputPtr, batchCount, batchLength, stream);
}