#include "Controller/ControllerEvents/BuildEvent.h"

BuildEvent::BuildEvent(const std::string& json, const bool cascadeMode) : passingJson(json), cascadeMode(cascadeMode), succeeded(true)
{
}

BuildEvent::~BuildEvent()
{
}

const std::string& BuildEvent::getJson()
{
	return this->passingJson;
}

const bool BuildEvent::isCascadeMode()
{
	return this->cascadeMode;
}

void BuildEvent::setSucceeded(const bool val)
{
	this->succeeded = val;
}

bool BuildEvent::getSucceeded()
{
	return this->succeeded;
}
