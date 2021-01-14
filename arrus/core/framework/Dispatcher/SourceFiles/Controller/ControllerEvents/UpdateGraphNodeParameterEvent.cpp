#include "Controller/ControllerEvents/UpdateGraphNodeParameterEvent.h"


UpdateGraphNodeParameterEvent::UpdateGraphNodeParameterEvent(const int nodeId, const std::string& parameterName, const VariableAnyValue parameterValue)
{
	this->nodeId = nodeId;
	this->parameterName = parameterName;
	this->parameterValue = parameterValue;
}


UpdateGraphNodeParameterEvent::~UpdateGraphNodeParameterEvent()
{
}

int UpdateGraphNodeParameterEvent::getNodeId()
{
	return this->nodeId;
}

const std::string& UpdateGraphNodeParameterEvent::getParameterName()
{
	return this->parameterName;
}

const VariableAnyValue& UpdateGraphNodeParameterEvent::getParameterValue()
{
	return this->parameterValue;
}