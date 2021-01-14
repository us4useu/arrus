#include "Model/GraphNodesLibrary/GraphNodes/PhasedArrayGraphNode/PhasedArrayGraphNode.h"
#include "Model/GraphNodesLibrary/GraphNodes/PhasedArrayGraphNode/CudaPhasedArrayGraphNode.cuh"

GraphNodesFactoryRegister<PhasedArrayGraphNode> PhasedArrayGraphNode::graphNodesFactoryRegister("phasedarray");

PhasedArrayGraphNode::PhasedArrayGraphNode()
{
	this->setNodeVariable("width", VariableAnyValue(256));
	this->setNodeVariable("height", VariableAnyValue(512));
	this->setNodeVariable("focusing", VariableAnyValue(false));
	this->setNodeVariable("openingAngle", VariableAnyValue(90.0f));
}

PhasedArrayGraphNode::~PhasedArrayGraphNode()
{
	this->releaseGPUMemory(&this->outputData);
	this->releaseGPUMemory(&this->anglesInfo);
	this->releaseGPUMemory(&this->focusesInfo);
}

void PhasedArrayGraphNode::process(cudaStream_t& defaultStream)
{
	int width = this->getNodeVariable("width").getValue<int>();
	int height = this->getNodeVariable("height").getValue<int>();
	this->allocGPUMemory<float>(&this->outputData, Dims(width, height));
	
	int samplesCount = this->inputData.getPtrProperty("samplesCount").getValue<int>();
	float receiverWidth = this->inputData.getPtrProperty("receiverWidth").getValue<float>();
	float soundVelocity = this->inputData.getPtrProperty("speedOfSound").getValue<float>();
	int receiversCount = this->inputData.getPtrProperty("numReceivers").getValue<int>();
	float samplingFrequency = this->inputData.getPtrProperty("samplingFrequency").getValue<float>();

	float areaHeight = samplesCount * soundVelocity / samplingFrequency * 0.5f;

	bool focusing = this->getNodeVariable("focusing").getValue<bool>();

	if (focusing)
	{
		std::vector<float> angles = this->inputData.getPtrProperty("steeringAngles").getValue<std::vector<float>>();
		if (angles.size() != width)
		{
			DISPATCHER_LOG(DispatcherLogType::ERROR_, std::string("Number of acquisitions must be equal to image width in phased array with focusing option enabled."));
			return;
		}
		this->allocGPUMemory<float>(&this->anglesInfo, (int)angles.size());
		CUDA_ASSERT(cudaMemcpyAsync(this->anglesInfo.getVoidPtr(), &angles[0], sizeof(float) * angles.size(), cudaMemcpyHostToDevice, defaultStream));

		CudaPhasedArrayGraphNode::phasedArrayWithFocusing(this->inputData.getPtr<float*>(), this->outputData.getPtr<float*>(), defaultStream,
			width, height, soundVelocity, areaHeight, receiverWidth, receiversCount, samplingFrequency,
			samplesCount, this->anglesInfo.getPtr<float*>());
	}
	else
	{
		float openingAngle = this->getNodeVariable("openingAngle").getValue<float>();
		std::vector<float2> focuses = this->inputData.getPtrProperty("focuses").getValue<std::vector<float2>>();
			
		this->allocGPUMemory<float>(&this->focusesInfo, (int)focuses.size() * 2);
		CUDA_ASSERT(cudaMemcpyAsync(this->focusesInfo.getVoidPtr(), &focuses[0], sizeof(float) * focuses.size() * 2, cudaMemcpyHostToDevice, defaultStream));

		CudaPhasedArrayGraphNode::phasedArray(this->inputData.getPtr<float*>(), this->outputData.getPtr<float*>(), defaultStream,
			width, height, soundVelocity, areaHeight, receiverWidth, receiversCount, samplingFrequency,
			samplesCount, openingAngle, (float2*)focusesInfo.getPtr<float*>(), (int)focuses.size());

	}
		
	this->outputData.copyExtraData(this->inputData);
}