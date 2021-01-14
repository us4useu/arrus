#include "Model/GraphNodesLibrary/GraphNodes/SaftZeroGraphNode/SaftZeroGraphNode.h"

GraphNodesFactoryRegister<SaftZeroGraphNode> SaftZeroGraphNode::graphNodesFactoryRegister("saftzero");

SaftZeroGraphNode::SaftZeroGraphNode()
{

}

SaftZeroGraphNode::~SaftZeroGraphNode()
{
	this->releaseGPUMemory(&this->outputData);
}

void SaftZeroGraphNode::process(cudaStream_t& defaultStream)
{
	this->allocGPUMemory<float>(&this->outputData, this->inputData.getDims());

	const int signalLen = this->inputData.getDims().x;
	const int apertureLen = this->inputData.getDims().y;
	const float soundVelocity = this->inputData.getPtrProperty("speedOfSound").getValue<float>();
	const float frequency = this->inputData.getPtrProperty("samplingFrequency").getValue<float>();
	const float pitch = this->inputData.getPtrProperty("pitch").getValue<float>();
	
	cudaSaftZeroGraphNode.saftZero(this->inputData.getPtr<float*>(), 
		this->outputData.getPtr<float*>(),
		defaultStream, 
		apertureLen, 
		signalLen,
		soundVelocity,
		frequency,
		pitch);

	this->outputData.copyExtraData(this->inputData);
}