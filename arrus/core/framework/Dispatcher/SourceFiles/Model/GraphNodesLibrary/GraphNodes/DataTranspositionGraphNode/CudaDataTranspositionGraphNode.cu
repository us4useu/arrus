#include "Model/GraphNodesLibrary/GraphNodes/DataTranspositionGraphNode/CudaDataTranspositionGraphNode.cuh"
#include "Model/GraphNodesLibrary/CudaUtils/CudaTranspositionUtils.cuh"

void CudaDataTranspositionGraphNode::transposeData(const short *inputPtr, float *outputPtr, const cudaStream_t& stream, 
												   const int xSize, const int ySize, const int zSize)
{
	CudaTranspositionUtils::transpose(inputPtr, outputPtr, xSize, ySize, stream, zSize);
}

void CudaDataTranspositionGraphNode::transposeData(const short *inputPtr, short *outputPtr, const cudaStream_t& stream,
												   const int xSize, const int ySize, const int zSize)
{
	CudaTranspositionUtils::transpose(inputPtr, outputPtr, xSize, ySize, stream, zSize);
}

void CudaDataTranspositionGraphNode::transposeData(const int *inputPtr, int *outputPtr, const cudaStream_t& stream,
												   const int xSize, const int ySize, const int zSize)
{
	CudaTranspositionUtils::transpose(inputPtr, outputPtr, xSize, ySize, stream, zSize);
}

void CudaDataTranspositionGraphNode::transposeData(const float *inputPtr, float *outputPtr, const cudaStream_t& stream,
												   const int xSize, const int ySize, const int zSize)
{
	CudaTranspositionUtils::transpose(inputPtr, outputPtr, xSize, ySize, stream, zSize);
}

void CudaDataTranspositionGraphNode::transposeData(const double *inputPtr, double *outputPtr, const cudaStream_t& stream,
												   const int xSize, const int ySize, const int zSize)
{
	CudaTranspositionUtils::transpose(inputPtr, outputPtr, xSize, ySize, stream, zSize);
}

void CudaDataTranspositionGraphNode::transposeData(const float2 *inputPtr, float2 *outputPtr, const cudaStream_t& stream,
												   const int xSize, const int ySize, const int zSize)
{
	CudaTranspositionUtils::transpose(inputPtr, outputPtr, xSize, ySize, stream, zSize);
}