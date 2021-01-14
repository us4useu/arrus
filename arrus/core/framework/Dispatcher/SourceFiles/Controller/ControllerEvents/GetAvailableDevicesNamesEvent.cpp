#include "Controller/ControllerEvents/GetAvailableDevicesNamesEvent.h"


GetAvailableDevicesNamesEvent::GetAvailableDevicesNamesEvent()
{

}


GetAvailableDevicesNamesEvent::~GetAvailableDevicesNamesEvent()
{
}

std::vector<std::string> GetAvailableDevicesNamesEvent::getDevicesNames()
{
	return this->devicesNames;
}

void GetAvailableDevicesNamesEvent::setDevicesNames(const std::vector<std::string>& devicesNames)
{
	this->devicesNames = devicesNames;
}