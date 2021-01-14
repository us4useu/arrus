#include "Model/GraphNodesLibrary/GraphNodes/HilbertGraphNode/CudaHilbertGraphNode.cuh"

__global__ void prepareFftInputData(cufftComplex *cufftData, const float *inputData)
{
	int idxx = threadIdx.x + blockDim.x * blockIdx.x;
	int idxy = threadIdx.y + blockDim.y * blockIdx.y;
	int sizex = blockDim.x * gridDim.x;
	int sizey = blockDim.y * gridDim.y;
	
	cufftComplex complex;
	complex.x = 0.0f;
	complex.y = 0.0f;

	if (idxy < (sizey >> 1))
		complex.x = inputData[idxx + idxy * sizex];

	cufftData[idxx + idxy * sizex] = complex;
}

__global__ void transformFftData(cufftComplex *cufftData, const int batchCount)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	cufftComplex complex;
	complex.x = 0.0f;
	complex.y = 0.0f;

	int allThreads = blockDim.x * gridDim.x;
	if (idx < batchCount)
	{
		complex = cufftData[idx];
	}
	else if (idx > (allThreads >> 1) + batchCount && idx < (allThreads >> 1) + 2 * batchCount)
	{
		complex = cufftData[idx];
	}
	else if (idx < ((allThreads >> 1)))
	{
		complex = cufftData[idx];
		complex.x *= 2.0f;
		complex.y *= 2.0f;
	}

	cufftData[idx] = complex;
}

__global__ void complexAbs(const cufftComplex *cufftData, float* outputData)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	cufftComplex complex = cufftData[idx];

	float result = sqrtf(complex.x * complex.x + complex.y * complex.y);

	outputData[idx] = result;
}

CudaHilbertGraphNode::CudaHilbertGraphNode()
{
	this->allocatedDataCount = 0;
}

CudaHilbertGraphNode::~CudaHilbertGraphNode()
{
	this->releaseMemory();
}

void CudaHilbertGraphNode::allocMemory(const int batchLength, const int batchCount)
{
	int dataCount = batchLength * batchCount * 2;
	if (dataCount != this->allocatedDataCount)
	{
		this->releaseMemory();

		int cufftEmbed = 1;
		int n = batchLength * 2;
		CUFFT_ASSERT(cufftPlanMany(&this->cufftPlan, 1, &n, 
			&cufftEmbed, batchCount, 1, 
			&cufftEmbed, batchCount, 1, 
			CUFFT_C2C, batchCount));
		CUDA_ASSERT(cudaMalloc((void **)&this->cufftData, sizeof(cufftComplex) * dataCount));
		this->allocatedDataCount = dataCount;
	}
}

void CudaHilbertGraphNode::releaseMemory()
{
	if (this->allocatedDataCount != 0)
	{
		CUFFT_ASSERT(cufftDestroy(this->cufftPlan));
		CUDA_ASSERT(cudaFree(this->cufftData));
	}
}

void CudaHilbertGraphNode::hilbertTransform(const float *inputData, float *outputData, const cudaStream_t& stream, const int batchLength, const int batchCount)
{
	this->allocMemory(batchLength, batchCount);

	dim3 block(16, 16);
	dim3 grid(batchCount / block.x, batchLength * 2 / block.y);
	prepareFftInputData << <grid, block, 0, stream >> >(this->cufftData, inputData);
	CUDA_ASSERT(cudaGetLastError());

	CUFFT_ASSERT(cufftSetStream(this->cufftPlan, stream));
	CUFFT_ASSERT(cufftExecC2C(this->cufftPlan, this->cufftData, this->cufftData, CUFFT_FORWARD));
	
	block = dim3(256);
	grid = dim3(batchLength * batchCount * 2 / block.x);
	transformFftData << <grid, block, 0, stream >> >(this->cufftData, batchCount);
	CUDA_ASSERT(cudaGetLastError());

	CUFFT_ASSERT(cufftExecC2C(this->cufftPlan, this->cufftData, this->cufftData, CUFFT_INVERSE));

	grid.x = grid.x / 2;
	complexAbs << <grid, block, 0, stream >> >(this->cufftData, outputData);
	CUDA_ASSERT(cudaGetLastError());
}