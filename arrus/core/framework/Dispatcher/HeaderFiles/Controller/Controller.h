#pragma once

#define BOOST_THREAD_PROVIDES_INTERRUPTIONS

#include <boost/thread.hpp>
#include <unordered_map>

#include "ControllerActions/ControllerAction.h"
#include "ControllerEvents/ControllerEvent.h"
#include "ConcurrentBlockingQueue.h"
#include "Model/Model.h"

class Controller {
private:
    ConcurrentBlockingQueue<std::shared_ptr < IControllerEvent>> eventsQueue;
    boost::thread controllerThread;
    std::unordered_map<unsigned int, std::shared_ptr<ControllerAction>> eventsActionsMap;
    std::shared_ptr <Model> model;
    bool isWorking;

    void work();

    void createEventsActionMap();

public:
    Controller();

    ~Controller();

    void destroyModel();

    std::shared_ptr <Model> getModel();

    void start();

    void stop();

    void stopWork();

    void kill();

    void synchronize();

    void sendControllerEvent(std::shared_ptr <IControllerEvent> controllerEvent);
};

