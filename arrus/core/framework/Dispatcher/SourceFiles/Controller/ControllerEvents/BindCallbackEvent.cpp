#include "Controller/ControllerEvents/BindCallbackEvent.h"

BindCallbackEvent::BindCallbackEvent(const int id, const boost::function<void(void* data, int iterationId, int graphNodeId, int dimX, int dimY, int dimZ, int dataType)> callback) :
	graphNodeId(id), callbackFunction(callback)
{
}

BindCallbackEvent::~BindCallbackEvent()
{
}

const int BindCallbackEvent::getNodeId()
{
	return graphNodeId;
}

const boost::function<void(void* data, int iterationId, int graphNodeId, int dimX, int dimY, int dimZ, int dataType)> BindCallbackEvent::getCallback()
{
	return callbackFunction;
}
