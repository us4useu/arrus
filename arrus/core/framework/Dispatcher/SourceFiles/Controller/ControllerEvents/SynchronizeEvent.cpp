#include "Controller/ControllerEvents/SynchronizeEvent.h"


SynchronizeEvent::SynchronizeEvent()
{
	// makes mutex locked on creation
	this->mutex.lock();
}


SynchronizeEvent::~SynchronizeEvent()
{
    this->mutex.unlock();
}

void SynchronizeEvent::wait()
{
	this->mutex.lock();
}

void SynchronizeEvent::notify()
{
	this->mutex.unlock();
}
