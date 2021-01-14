#include "Controller/Controller.h"

#include "Controller/ControllerEvents/BuildEvent.h"
#include "Controller/ControllerActions/BuildAction/BuildAction.h"
#include "Controller/ControllerEvents/BindCallbackEvent.h"
#include "Controller/ControllerActions/BindCallbackAction.h"
#include "Controller/ControllerEvents/StopControllerEvent.h"
#include "Controller/ControllerActions/StopControllerAction.h"
#include "Controller/ControllerActions/StartModelAction.h"
#include "Controller/ControllerEvents/StartModelEvent.h"
#include "Controller/ControllerActions/StopModelAction.h"
#include "Controller/ControllerEvents/StopModelEvent.h"
#include "Controller/ControllerEvents/SynchronizeEvent.h"
#include "Controller/ControllerActions/SynchronizeAction.h"
#include "Controller/ControllerActions/SetUserInputDataAction.h"
#include "Controller/ControllerEvents/SetUserInputDataEvent.h"
#include "Controller/ControllerActions/UpdateGraphNodeParameterAction.h"
#include "Controller/ControllerEvents/UpdateGraphNodeParameterEvent.h"
#include "Controller/ControllerActions/GetAvailableDevicesNamesAction.h"
#include "Controller/ControllerEvents/GetAvailableDevicesNamesEvent.h"
#include "Controller/ControllerActions/SetDeviceAction.h"
#include "Controller/ControllerEvents/SetDeviceEvent.h"

Controller::Controller()
{
	this->isWorking = false;
	this->createEventsActionMap();
}


Controller::~Controller()
{
}

void Controller::createEventsActionMap()
{
	this->eventsActionsMap.insert(std::make_pair(BuildEvent::getStaticUniqueID(), std::shared_ptr<BuildAction>(new BuildAction(this))));
	this->eventsActionsMap.insert(std::make_pair(StopControllerEvent::getStaticUniqueID(), std::shared_ptr<StopControllerAction>(new StopControllerAction(this))));
	this->eventsActionsMap.insert(std::make_pair(BindCallbackEvent::getStaticUniqueID(), std::shared_ptr<BindCallbackAction>(new BindCallbackAction(this))));
	this->eventsActionsMap.insert(std::make_pair(StartModelEvent::getStaticUniqueID(), std::shared_ptr<StartModelAction>(new StartModelAction(this))));
	this->eventsActionsMap.insert(std::make_pair(StopModelEvent::getStaticUniqueID(), std::shared_ptr<StopModelAction>(new StopModelAction(this))));
	this->eventsActionsMap.insert(std::make_pair(SynchronizeEvent::getStaticUniqueID(), std::shared_ptr<SynchronizeAction>(new SynchronizeAction(this))));
	this->eventsActionsMap.insert(std::make_pair(SetUserInputDataEvent<short>::getStaticUniqueID(), std::shared_ptr<SetUserInputDataAction<short>>(new SetUserInputDataAction<short>(this))));
	this->eventsActionsMap.insert(std::make_pair(SetUserInputDataEvent<int>::getStaticUniqueID(), std::shared_ptr<SetUserInputDataAction<int>>(new SetUserInputDataAction<int>(this))));
	this->eventsActionsMap.insert(std::make_pair(SetUserInputDataEvent<float>::getStaticUniqueID(), std::shared_ptr<SetUserInputDataAction<float>>(new SetUserInputDataAction<float>(this))));
	this->eventsActionsMap.insert(std::make_pair(SetUserInputDataEvent<double>::getStaticUniqueID(), std::shared_ptr<SetUserInputDataAction<double>>(new SetUserInputDataAction<double>(this))));
	this->eventsActionsMap.insert(std::make_pair(SetUserInputDataEvent<float2>::getStaticUniqueID(), std::shared_ptr<SetUserInputDataAction<float2>>(new SetUserInputDataAction<float2>(this))));
	this->eventsActionsMap.insert(std::make_pair(UpdateGraphNodeParameterEvent::getStaticUniqueID(), std::shared_ptr<UpdateGraphNodeParameterAction>(new UpdateGraphNodeParameterAction(this))));
	this->eventsActionsMap.insert(std::make_pair(GetAvailableDevicesNamesEvent::getStaticUniqueID(), std::shared_ptr<GetAvailableDevicesNamesAction>(new GetAvailableDevicesNamesAction(this))));
	this->eventsActionsMap.insert(std::make_pair(SetDeviceEvent::getStaticUniqueID(), std::shared_ptr<SetDeviceAction>(new SetDeviceAction(this))));
}

std::shared_ptr<Model> Controller::getModel()
{
	if (this->model == nullptr)
		this->model = std::shared_ptr<Model>(new Model());

	return this->model;
}

void Controller::destroyModel()
{
	this->model.reset();
}

void Controller::start()
{
	this->isWorking = true;
	this->controllerThread = boost::thread(&Controller::work, this);
}

void Controller::work()
{
	while (this->isWorking)
	{
		std::shared_ptr<IControllerEvent> currentProcessedEvent = this->eventsQueue.pop();
		std::shared_ptr<ControllerAction> controllerAction = this->eventsActionsMap[currentProcessedEvent->getUniqueID()];
		controllerAction->performAction(currentProcessedEvent);
	}
}

void Controller::stop()
{	
	this->sendControllerEvent(std::shared_ptr<StopControllerEvent>(new StopControllerEvent()));
	this->controllerThread.join();
}

void Controller::stopWork()
{
	this->isWorking = false;
}

void Controller::sendControllerEvent(std::shared_ptr<IControllerEvent> controllerEvent)
{
	this->eventsQueue.push(controllerEvent);
}

void Controller::synchronize()
{
	std::shared_ptr<SynchronizeEvent> syncEvent(new SynchronizeEvent());
	this->sendControllerEvent(syncEvent);
	syncEvent->wait();
}

void Controller::kill()
{
	this->model->kill();
	this->controllerThread.interrupt();
}
