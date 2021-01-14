#include "Controller/ControllerEvents/StartModelEvent.h"


StartModelEvent::StartModelEvent(const bool startOnce)
{
	this->startOnce = startOnce;
}


StartModelEvent::~StartModelEvent()
{
}

const bool StartModelEvent::doesStartOnce()
{
	return this->startOnce;
}