#include "Model/GraphNodesLibrary/GraphNodes/QuadratureDemodulationGraphNode/CudaQuadratureDemodulationGraphNode.cuh"
#include "math_constants.h"

__global__ void gpuRfToIq(const float *inputRfPtr, float2 *outputIqPtr, const float samplingFrequency, const float transmitFrequency, 
						  const int batchLength, const int maxThreadId, const int startSampleNumber)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= maxThreadId)
		return;

	float rfSample = inputRfPtr[idx];
	int sampleNumber = idx % batchLength;

	float cosinus, sinus;
	__sincosf(2.0f * CUDART_PI_F * transmitFrequency / samplingFrequency * (sampleNumber + startSampleNumber), &sinus, &cosinus);

	float2 iq;
	iq.x = 2.0f * rfSample * cosinus;
	iq.y = 2.0f * rfSample * sinus;

	outputIqPtr[idx] = iq;
}

__global__ void gpuDecimation(const float2* inputIqPtr, float2* outputIqPtr, const int batchLength, const int maxThreadId, const int decimationValue)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= maxThreadId)
		return;

	int decimatedBatchLength = (int)ceilf((float)batchLength / decimationValue);

	outputIqPtr[idx] = inputIqPtr[(idx / decimatedBatchLength) * batchLength + (idx % decimatedBatchLength) * decimationValue];
}

void CudaQuadratureDemodulationGraphNode::rfToIq(const float *inputRfPtr, float2 *outputIqPtr, const cudaStream_t& stream, const int batchLength,
												 const int batchCount, const float samplingFrequency, const float transmitFrequency, const int startSampleNumber)
{
	dim3 block(512);
	int dataCount = batchLength * batchCount;
	dim3 grid((dataCount + block.x - 1) / block.x);
	gpuRfToIq << <grid, block, 0, stream >> >(inputRfPtr, outputIqPtr, samplingFrequency, transmitFrequency, batchLength, dataCount, startSampleNumber);
	CUDA_ASSERT(cudaGetLastError());
}

void CudaQuadratureDemodulationGraphNode::decimation(const float2* inputIqPtr, float2* outputIqPtr, const cudaStream_t& stream, const int batchLength,
													 const int batchCount, const int decimationValue)
{
	dim3 block(512);
	int threadsCount = (int)ceilf((float)batchLength / decimationValue) * batchCount;
	dim3 grid((threadsCount + block.x - 1) / block.x);
	gpuDecimation << <grid, block, 0, stream >> >(inputIqPtr, outputIqPtr, batchLength, threadsCount, decimationValue);
	CUDA_ASSERT(cudaGetLastError());
}