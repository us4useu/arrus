#include "Model/GraphNodesLibrary/GraphNodes/PhasedArrayGraphNode/CudaPhasedArrayGraphNode.cuh"
#include "Model/GraphNodesLibrary/CudaUtils/CudaCommonsUtils.cuh"
#include <math_constants.h>
#include <algorithm>

__global__ void gpuPhasedArray(const float *input, float *output, const int resultHeightInPixels, const int resultWidthInPixels, const float areaHeight,
	const int receiversCount, const float receiverWidth, const float soundVelocity, const float samplingFrequency, const int samplesCount, 
	const float openingAngle, const float2* focusesInfo, const int focusesNumber)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	extern __shared__ float2 sharedFocuses[];
	for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < focusesNumber; i += blockDim.x * blockDim.y)
		sharedFocuses[i] = focusesInfo[i];
	__syncthreads();

	float result = 0.0f;

	// distance from transmit point
	float radius = float(y) / float(resultHeightInPixels - 1) * areaHeight;
	// transmit angle for current line
	float currAngle = (-openingAngle * 0.5f) + float(x) * openingAngle / float(resultWidthInPixels - 1);
	float currAngleRadians = CudaCommonsUtils::radians(currAngle);

	// vertical distance from transmit point
	float heightInCone = radius * cosf(currAngleRadians);
	// horizontal distance from image center
	float horizontalWidthFromCenter = radius * sinf(currAngleRadians);

	for (int f = 0; f < focusesNumber; ++f)
	{
		float2 currFocus = sharedFocuses[f];
		int firstTransmitter = CudaCommonsUtils::clamp(currFocus.x + 0.5f * receiverWidth, 0.0f, receiverWidth) * float(receiversCount - 1) / receiverWidth;
		float firstTransmitterRealPosition = ((float(firstTransmitter) / float(receiversCount - 1) * receiverWidth) - (receiverWidth * 0.5f));
		float transmitHardwareDelay = sqrtf(CudaCommonsUtils::square(firstTransmitterRealPosition - currFocus.x) + CudaCommonsUtils::square(currFocus.y));
		//float transmitHardwareDelay = sqrtf(CudaCommonsUtils::square(CudaCommonsUtils::clamp(currFocus.x, -0.5f * receiverWidth, 0.5f * receiverWidth) - currFocus.x) +
		//									CudaCommonsUtils::square(currFocus.y));
		float transmitDistance = sqrtf(CudaCommonsUtils::square(horizontalWidthFromCenter - currFocus.x) +
									   CudaCommonsUtils::square(heightInCone - currFocus.y)) - transmitHardwareDelay;
		const float* currInput = input + receiversCount * samplesCount * f;
		for (int r = 0; r < receiversCount; ++r)
		{
			float realReceiver = float(r) / float(receiversCount - 1) * receiverWidth;
			float realReceiverFromCenter = (realReceiver - (receiverWidth * 0.5f));

			float distance = sqrtf(CudaCommonsUtils::square(horizontalWidthFromCenter - realReceiverFromCenter) +
								   CudaCommonsUtils::square(heightInCone)) + transmitDistance;

			float T = distance / soundVelocity;
			int sample = T * samplingFrequency;
			int idx = samplesCount * r + sample;

			if ((sample < samplesCount) && (sample >= 0))
			{
				result += currInput[idx];
			}
		}
	}

	output[x + y * blockDim.x * gridDim.x] = result;
}

__global__ void gpuPhasedArrayWithFocusing(const float *input, float *output, const int resultWidthInPixels, const float soundVelocity,
	const float areaHeight, const float receiverWidth, const int receiversCount, const float samplingFrequency, const int samplesCount, 
	const float* transmitAngles)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// distance from transmit point
	float radius = float(y) / float(resultWidthInPixels - 1) * areaHeight;
	// transmit angle for current line
	float currAngle = transmitAngles[x];
	float currAngleRadians = CudaCommonsUtils::radians(currAngle);
	// vertical distance from transmit point
	float heightInCone = radius * cosf(currAngleRadians);
	// horizontal distance from image center
	float horizontalWidthFromCenter = radius * sinf(currAngleRadians);

	float result = 0.0f;

	const float *currInput = input + receiversCount * samplesCount * x;

	for (int r = 0; r < receiversCount; ++r)
	{
		float realReceiver = float(r) / float(receiversCount - 1) * receiverWidth;
		float realReceiverFromCenter = realReceiver - (receiverWidth * 0.5f);

		float distance = radius + sqrtf(CudaCommonsUtils::square(horizontalWidthFromCenter - realReceiverFromCenter) + CudaCommonsUtils::square(heightInCone));

		float T = distance / soundVelocity;
		int sample = T * samplingFrequency;
		int idx = samplesCount * r + sample;

		if ((sample < samplesCount) && (sample >= 0))
		{
			result += currInput[idx];
		}
	}
	output[x + y * blockDim.x * gridDim.x] = result;
}

void CudaPhasedArrayGraphNode::phasedArray(const float *input, float *output, const cudaStream_t& stream, const int resultWidthInPixels,
	const int resultHeightInPixels, const float soundVelocity, const float areaHeight, const float receiverWidth, const int receiversCount, 
	const float samplingFrequency, const int samplesCount, const float openingAngle, const float2* focusesInfo, const int focusesNumber)
{
	dim3 blockDim(16, 16);
	dim3 gridDim(resultWidthInPixels / blockDim.x, resultHeightInPixels / blockDim.y);
	int externalMemory = sizeof(float2) * focusesNumber;
	gpuPhasedArray << <gridDim, blockDim, externalMemory, stream >> >(input, output, resultHeightInPixels, resultWidthInPixels, areaHeight,
		receiversCount, receiverWidth, soundVelocity, samplingFrequency, samplesCount, openingAngle, focusesInfo, focusesNumber);
	CUDA_ASSERT(cudaGetLastError());
}

void CudaPhasedArrayGraphNode::phasedArrayWithFocusing(const float *input, float *output, const cudaStream_t& stream, const int resultWidthInPixels,
	const int resultHeightInPixels, const float soundVelocity, const float areaHeight, const float receiverWidth, const int receiversCount, 
	const float samplingFrequency, const int samplesCount, const float* transmitAngles)
{
	dim3 blockDim(1, std::min(512, resultHeightInPixels));
	dim3 gridDim(resultWidthInPixels / blockDim.x, resultHeightInPixels / blockDim.y);
	gpuPhasedArrayWithFocusing << <gridDim, blockDim, 0, stream >> >(input, output, resultHeightInPixels, soundVelocity,
		areaHeight, receiverWidth, receiversCount, samplingFrequency, samplesCount, transmitAngles);
	CUDA_ASSERT(cudaGetLastError());
}