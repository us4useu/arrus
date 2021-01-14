#include "Model/GraphNodesLibrary/GraphNodes/DataTranspositionGraphNode/DataTranspositionGraphNode.h"

GraphNodesFactoryRegister<DataTranspositionGraphNode> DataTranspositionGraphNode::graphNodesFactoryRegister("dataTransposition");

DataTranspositionGraphNode::DataTranspositionGraphNode()
{
}


DataTranspositionGraphNode::~DataTranspositionGraphNode()
{
	this->releaseGPUMemory(&this->outputData);
}

void DataTranspositionGraphNode::process(cudaStream_t& defaultStream)
{
        boost::variant<short*, int*, float*, double*, float2*>&& ptr = this->inputData.getRawPtr();
        boost::apply_visitor(InputDataTranspositionVisitator(this, defaultStream, this->inputData.getPtrProperty("rawDataFromHAL").getValue<bool>()), ptr);

	this->outputData.copyExtraData(this->inputData);
	this->outputData.setPtrProperty("rawDataFromHAL", VariableAnyValue(false));
}
