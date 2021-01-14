#include "Model/GraphNodesLibrary/GraphNodes/ConeShapedTransformationGraphNode/ConeShapedTransformationGraphNode.h"
#include "Model/GraphNodesLibrary/GraphNodes/ConeShapedTransformationGraphNode/CudaConeShapedTransformationGraphNode.cuh"

GraphNodesFactoryRegister<ConeShapedTransformationGraphNode> ConeShapedTransformationGraphNode::graphNodesFactoryRegister("coneShapedTransformation");

ConeShapedTransformationGraphNode::ConeShapedTransformationGraphNode()
{
	this->setNodeVariable("outputWidth", VariableAnyValue(512));
	this->setNodeVariable("outputHeight", VariableAnyValue(512));
	this->setNodeVariable("openingAngle", VariableAnyValue(90.0f));
	this->setNodeVariable("outputWidthInMeters", VariableAnyValue(0.061f));
	this->setNodeVariable("outputHeightInMeters", VariableAnyValue(0.061f));
}


ConeShapedTransformationGraphNode::~ConeShapedTransformationGraphNode()
{
	this->releaseGPUMemory(&this->outputData);
}

void ConeShapedTransformationGraphNode::process(cudaStream_t& defaultStream)
{
	int outputWidth = this->getNodeVariable("outputWidth").getValue<int>();
	int outputHeight = this->getNodeVariable("outputHeight").getValue<int>();
	this->allocGPUMemory<float>(&this->outputData, Dims(outputWidth, outputHeight));
	
	int samplesCount = this->inputData.getPtrProperty("samplesCount").getValue<int>();
	float soundVelocity = this->inputData.getPtrProperty("speedOfSound").getValue<float>();
	float samplingFrequency = this->inputData.getPtrProperty("samplingFrequency").getValue<float>();
	float areaHeight = samplesCount * soundVelocity / samplingFrequency * 0.5f;

	int inputWidth = this->inputData.getDims().x;
	int inputHeight = this->inputData.getDims().y;
	float openingAngle = this->getNodeVariable("openingAngle").getValue<float>();
	float outputWidthInMeters = this->getNodeVariable("outputWidthInMeters").getValue<float>();
	float outputHeightInMeters = this->getNodeVariable("outputHeightInMeters").getValue<float>();

	CudaConeShapedTransformationGraphNode::coneShapedTransformation(this->inputData.getPtr<float*>(), this->outputData.getPtr<float*>(), defaultStream,
		inputWidth, inputHeight, openingAngle, outputWidthInMeters, outputHeightInMeters, areaHeight, outputWidth, outputHeight);

	this->outputData.copyExtraData(this->inputData);
}