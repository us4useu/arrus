#include "Model/GraphNodesLibrary/GraphNodes/QuadratureDemodulationGraphNode/QuadratureDemodulationGraphNode.h"
#include "Model/GraphNodesLibrary/GraphNodes/QuadratureDemodulationGraphNode/CudaQuadratureDemodulationGraphNode.cuh"
#include "Utils/DispatcherLogger.h"
#include <cmath>

GraphNodesFactoryRegister<QuadratureDemodulationGraphNode> QuadratureDemodulationGraphNode::graphNodesFactoryRegister("quadDemod");

QuadratureDemodulationGraphNode::QuadratureDemodulationGraphNode()
{
	this->setNodeVariable("decimation", VariableAnyValue(1));
	this->setNodeVariable("feedforwardCoefficients", VariableAnyValue(std::string("")));
	this->setNodeVariable("feedbackCoefficients", VariableAnyValue(std::string("")));
}

QuadratureDemodulationGraphNode::~QuadratureDemodulationGraphNode()
{
	this->releaseGPUMemory(&this->iqBuffer);
}

void QuadratureDemodulationGraphNode::process(cudaStream_t& defaultStream)
{
	this->allocGPUMemory<float2>(&this->iqBuffer, this->inputData.getDims());
	int batchLength = this->inputData.getDims().x;
	int batchCount = this->inputData.getDims().y * this->inputData.getDims().z;

	float samplingFrequency = this->inputData.getPtrProperty("samplingFrequency").getValue<float>();
	float transmitFrequency = this->inputData.getPtrProperty("transmitFrequencies").getValue<std::vector<float>>()[0];

	float startDepth = this->inputData.getPtrProperty("startDepth").getValue<float>();
	float soundVelocity = this->inputData.getPtrProperty("speedOfSound").getValue<float>();
	int startSampleNumber = round(startDepth / soundVelocity * samplingFrequency * 2.0f);

	CudaQuadratureDemodulationGraphNode::rfToIq(this->inputData.getPtr<float*>(), this->iqBuffer.getPtr<float2*>(), defaultStream,
												batchLength, batchCount, samplingFrequency, transmitFrequency, startSampleNumber);

	this->iqBuffer.copyExtraData(this->inputData);
	this->iqBuffer.setPtrProperty("iq", VariableAnyValue(true));

	this->filter1DGraphNode.setInputData(this->iqBuffer);
	this->filter1DGraphNode.setNodeVariable("feedforwardCoefficients", this->getNodeVariable("feedforwardCoefficients"));
	this->filter1DGraphNode.setNodeVariable("feedbackCoefficients", this->getNodeVariable("feedbackCoefficients"));
	this->filter1DGraphNode.process(defaultStream);

	int decimationValue = this->getNodeVariable("decimation").getValue<int>();
	if (decimationValue == 1)
		this->outputData = this->filter1DGraphNode.getOutputData();
	else
	{
		int samplesCount = (unsigned int)ceilf(this->inputData.getDims().x / (float)decimationValue);
		this->iqBuffer.setDims(Dims(samplesCount, this->inputData.getDims().y, this->inputData.getDims().z));
		CudaQuadratureDemodulationGraphNode::decimation(this->filter1DGraphNode.getOutputData().getPtr<float2*>(), this->iqBuffer.getPtr<float2*>(), defaultStream,
														batchLength, batchCount, decimationValue);
		this->iqBuffer.copyExtraData(this->filter1DGraphNode.getOutputData());
		this->iqBuffer.setPtrProperty("samplesCount", VariableAnyValue(samplesCount));
		this->iqBuffer.setPtrProperty("samplingFrequency", VariableAnyValue(samplingFrequency / decimationValue));
		this->outputData = this->iqBuffer;
	}
}