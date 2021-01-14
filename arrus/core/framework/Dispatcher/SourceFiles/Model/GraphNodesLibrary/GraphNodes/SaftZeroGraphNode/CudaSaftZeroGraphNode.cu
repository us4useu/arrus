#include "Model/GraphNodesLibrary/GraphNodes/SaftZeroGraphNode/CudaSaftZeroGraphNode.cuh"
#include "Utils/linspace.hpp"
#include "Utils/sgn.hpp"
#include <vector>

/*
 * 1) Real to Complex transform
 * 2) Proper layout of data from:
 * /--------\
 * |11111122|
 * |22223333|
   |33      |
 * |        |
 * \--------/
 * into:
 * /--------\
 * |111111  |
 * |222222  |
 * |333333  |
 * |        |
 * \--------/
 */
__global__ void prepareSaftZeroInputData(const float *inputData, cufftComplex *cufftData, 
	const int apertureLen, const int signalLen)
{
	int idxx = threadIdx.x + blockDim.x * blockIdx.x;
	int idxy = threadIdx.y + blockDim.y * blockIdx.y;
	int sizex = blockDim.x * gridDim.x;
	//int sizey = blockDim.y * gridDim.y;

	cufftComplex complex;
	complex.x = 0.0f;
	complex.y = 0.0f;

	if (idxx < signalLen && idxy < apertureLen)
	{
		complex.x = inputData[idxx + idxy * signalLen];
	}

	cufftData[idxx + idxy * sizex] = complex;
}

__global__ void fftShift2D(cufftComplex *cufftData)
{
	int idxx = blockIdx.x * blockDim.x + threadIdx.x;
	int idxy = blockIdx.y * blockDim.y + threadIdx.y;
	int sizex = gridDim.x * blockDim.x;
	int sizey = gridDim.y * blockDim.y;

	// Arbitrary axis
	if (idxx < sizex / 2)
	{
		int idxxm = (sizex / 2 + idxx) % sizex;
		int idxym = (sizey / 2 + idxy) % sizey;
		int index1 = idxx  + sizex * idxy;
		int index2 = idxxm + sizex * idxym;

		cufftComplex reg1 = cufftData[index1];
		cufftComplex reg2 = cufftData[index2];

		cufftData[index2] = reg1;
		cufftData[index1] = reg2;
	}
}

__global__ void transformSaftZeroData(cufftComplex *inData, cufftComplex *outData, float *fkz, float maxFreq)
{

	int idxx = threadIdx.x + blockDim.x * blockIdx.x;
	int idxy = threadIdx.y + blockDim.y * blockIdx.y;
	int sizex = blockDim.x * gridDim.x;
	
	cufftComplex complex1 = { 0.0f, 0.0f };
	cufftComplex complex2;

	float newFreq = fkz[idxx + idxy * sizex];

	// sizex is length of signal
	float newIndex = (float)(newFreq / maxFreq / 2 * sizex);
	newIndex += sizex / 2;

	if (newIndex >= 0.0f && newIndex < sizex)
	{
		int sampleLin1 = __float2int_rd(newIndex);
		int sampleLin2 = __float2int_ru(newIndex);
		float sampleFrac = newIndex - truncf(newIndex);
		
		complex1 = inData[sampleLin1 + idxy * sizex];
		complex2 = inData[sampleLin2 + idxy * sizex];
		complex1.x *= 1.0f - sampleFrac;
		complex1.y *= 1.0f - sampleFrac;
		complex2.x *= sampleFrac;
		complex2.y *= sampleFrac;
		complex1.x += complex2.x;
		complex1.y += complex2.y;
	}
	
	outData[idxx + idxy * sizex] = complex1;
}

/*
 * 1) Complex to Real (Abs of real and imag)
 * 2) Proper data layout
 */
__global__ void complexSaftZeroAbs(const cufftComplex *cufftData, float* outputData,
	const int apertureLen, const int signalLen)
{
	int idxx = threadIdx.x + blockDim.x * blockIdx.x;
	int idxy = threadIdx.y + blockDim.y * blockIdx.y;
	int sizex = blockDim.x * gridDim.x;

	if (idxx < signalLen && idxy < apertureLen)
	{
		cufftComplex complex = cufftData[idxx + idxy * sizex];
		// No need to normalize
		//complex.x /= sizex * sizey;
		//complex.y /= sizex * sizey;

		float result = sqrtf(complex.x * complex.x + complex.y * complex.y);

		outputData[idxx + idxy * signalLen] = powf(result, 0.7f);
	}
}

CudaSaftZeroGraphNode::CudaSaftZeroGraphNode()
{
	this->allocatedApertureLen = 0;
	this->allocatedSignalLen = 0;
	this->fftApertureLen = 0;
	this->fftSignalLen = 0;
}

CudaSaftZeroGraphNode::~CudaSaftZeroGraphNode()
{
	this->releaseStructures();
}

void CudaSaftZeroGraphNode::allocStructures(const int apertureLen, 
	const int signalLen, 
	const float soundVelocity,
	const float frequency,
	const float pitch)
{
	if (apertureLen != this->allocatedApertureLen ||
		signalLen != this->allocatedSignalLen)
	{
		this->releaseStructures();

		// Next power of 2
		this->fftSignalLen = (int)pow(2.0f, (int)ceilf(log2((float)signalLen)));
		this->fftApertureLen = (int)pow(2.0f, (int)ceilf(log2((float)apertureLen)));

		CUFFT_ASSERT(cufftPlan2d(&this->cufftPlan,
			this->fftApertureLen,
			this->fftSignalLen,
			CUFFT_C2C));

		CUDA_ASSERT(cudaMalloc((void **)&this->cufftData,
			sizeof(cufftComplex) * this->fftApertureLen * this->fftSignalLen));

		CUDA_ASSERT(cudaMalloc((void **)&this->cufftOutData,
			sizeof(cufftComplex) * this->fftApertureLen * this->fftSignalLen));

		CUDA_ASSERT(cudaMalloc((void **)&this->devFkz,
			sizeof(float) * this->fftApertureLen * this->fftSignalLen));

		std::vector<float> fkz = std::vector<float>();
		fkz.reserve(this->fftApertureLen * this->fftSignalLen);

		float ERMv = soundVelocity / sqrtf(2.0f);
		
		std::vector<float> f = linspace(-frequency / 2, ((this->fftSignalLen / 2 - 1)*frequency / this->fftSignalLen), this->fftSignalLen);
		std::vector<float> kx = linspace(-1.0f/pitch / 2, ((this->fftApertureLen / 2 - 1)/pitch / this->fftApertureLen), this->fftApertureLen);

		for (int a = 0; a < this->fftApertureLen; a++)
		{
			for (int s = 0; s < this->fftSignalLen; s++)
			{
				fkz.push_back(ERMv * sgn(f[s]) *sqrt((kx[a] * kx[a]) + (f[s] * f[s]) / (ERMv * ERMv)) );
			}
		}
		
		CUDA_ASSERT(cudaMemcpy(this->devFkz,
			(void*)&fkz[0],
			sizeof(float) * this->fftApertureLen * this->fftSignalLen,
			cudaMemcpyHostToDevice));

		f.clear();
		kx.clear();
		fkz.clear();

		this->allocatedApertureLen = apertureLen;
		this->allocatedSignalLen = signalLen;
	}
}

void CudaSaftZeroGraphNode::releaseStructures()
{
	if (this->allocatedApertureLen != 0 &&
		this->allocatedSignalLen != 0)
	{
		CUFFT_ASSERT(cufftDestroy(this->cufftPlan));
		CUDA_ASSERT(cudaFree(this->cufftData));
		CUDA_ASSERT(cudaFree(this->cufftOutData));
		CUDA_ASSERT(cudaFree(this->devFkz));
	}
}

void CudaSaftZeroGraphNode::saftZero(const float *inputPtr, 
	float *outputPtr, 
	const cudaStream_t& stream,
	const int apertureLen, 
	const int signalLen,
	const float soundVelocity,
	const float frequency,
	const float pitch)
{
	this->allocStructures(apertureLen, signalLen, soundVelocity, frequency, pitch);

	dim3 block(16, 16);
	dim3 grid(this->fftSignalLen / block.x, this->fftApertureLen / block.y);
	
	prepareSaftZeroInputData << <grid, block, 0, stream >> >(inputPtr, this->cufftData, apertureLen, signalLen);
	CUDA_ASSERT(cudaGetLastError());

	CUFFT_ASSERT(cufftSetStream(this->cufftPlan, stream));
	CUFFT_ASSERT(cufftExecC2C(this->cufftPlan, this->cufftData, this->cufftData, CUFFT_FORWARD));

	fftShift2D << <grid, block, 0, stream >> >(this->cufftData);
	CUDA_ASSERT(cudaGetLastError());

	transformSaftZeroData << <grid, block, 0, stream >> >(this->cufftData, this->cufftOutData, this->devFkz, frequency / 2);
	CUDA_ASSERT(cudaGetLastError());

	fftShift2D << <grid, block, 0, stream >> >(this->cufftOutData);
	CUDA_ASSERT(cudaGetLastError());

	CUFFT_ASSERT(cufftExecC2C(this->cufftPlan, this->cufftOutData, this->cufftOutData, CUFFT_INVERSE));

	complexSaftZeroAbs << <grid, block, 0, stream >> >(this->cufftOutData, outputPtr, apertureLen, signalLen);
	CUDA_ASSERT(cudaGetLastError());
}
