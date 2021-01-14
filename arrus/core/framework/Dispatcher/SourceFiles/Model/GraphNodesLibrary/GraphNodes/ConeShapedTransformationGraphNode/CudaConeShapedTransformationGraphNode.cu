#include "Model/GraphNodesLibrary/GraphNodes/ConeShapedTransformationGraphNode/CudaConeShapedTransformationGraphNode.cuh"
#include "Model/GraphNodesLibrary/CudaUtils/CudaCommonsUtils.cuh"

__global__ void gpuConeShapedTransformation(const float* input, float *result, const int inputWidthInPixels, const int inputHeightInPixels,
	const float openingAngle, const float outputWidthInMeters, const float outputHeightInMeters, const float areaHeight, const int outputWidthInPixels,
	const int outputHeightInPixels)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float pixel = 0.0f;
	float heightDist = ((float)y / (float)outputHeightInPixels) * outputHeightInMeters;
	// horizontal distance from image center
	float middleDist = ((float)x - ((float)outputWidthInPixels * 0.5f)) * outputWidthInMeters / (float)outputWidthInPixels;
	float pointAngle = CudaCommonsUtils::degrees(atanf(fabs(middleDist) / heightDist));

	// if pixel angle is lower than transmit angle
	if (pointAngle < (openingAngle * 0.5f))
	{
		float distance = sqrtf(CudaCommonsUtils::square(middleDist) + CudaCommonsUtils::square(heightDist));
		if (distance <= areaHeight)
		{
			float mappedY = distance / areaHeight * (float)inputHeightInPixels;
			float angleShare;
			if (middleDist < 0.0f)
				angleShare = (openingAngle * 0.5f) - pointAngle;
			else
				angleShare = (openingAngle * 0.5f) + pointAngle;
			float mappedX = (angleShare / openingAngle) * (float)inputWidthInPixels;
			pixel = CudaCommonsUtils::getInterpolated2DValue(input, mappedX, mappedY, inputWidthInPixels, inputHeightInPixels);
		}
	}

	result[x + y * blockDim.x * gridDim.x] = pixel;
}

void CudaConeShapedTransformationGraphNode::coneShapedTransformation(const float *input, float *output, const cudaStream_t& stream, const int inputWidthInPixels,
	const int inputHeightInPixels, const float openingAngle, const float outputWidthInMeters, const float outputHeightInMeters, const float areaHeight, const int outputWidthInPixels,
	const int outputHeightInPixels)
{

	dim3 blockDim(16, 16);
	dim3 gridDim(outputWidthInPixels / blockDim.x, outputHeightInPixels / blockDim.y);
	gpuConeShapedTransformation << <gridDim, blockDim, 0, stream >> >(input, output, inputWidthInPixels, inputHeightInPixels, openingAngle,
		outputWidthInMeters, outputHeightInMeters, areaHeight, outputWidthInPixels, outputHeightInPixels);
	CUDA_ASSERT(cudaGetLastError());
}