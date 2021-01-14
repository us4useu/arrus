#include "Controller/ControllerEvents/SetDeviceEvent.h"


SetDeviceEvent::SetDeviceEvent(const std::string& deviceName)
{
	this->deviceName = deviceName;
}


SetDeviceEvent::~SetDeviceEvent()
{
}

std::string& SetDeviceEvent::getDeviceName()
{
	return this->deviceName;
}