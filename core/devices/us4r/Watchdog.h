#ifndef ARRUS_CORE_DEVICES_US4R_WATCHDOG_H
#define ARRUS_CORE_DEVICES_US4R_WATCHDOG_H

#include <functional>
#include <utility>
#include <mutex>
#include <ratio>
#include <arrus/common/asserts.h>

#include "arrus/core/common/logging.h"

namespace arrus::devices {

class Watchdog {
public:

    Watchdog()
        : logger{getLoggerFactory()->getLogger()} {
        INIT_ARRUS_DEVICE_LOGGER(logger, "Watchdog");
    }

    Watchdog(long long int timeout, std::function<bool()> callback)
        : logger{getLoggerFactory()->getLogger()},
          timeout(timeout), callback(std::move(callback)) {
        INIT_ARRUS_DEVICE_LOGGER(logger, "Watchdog");
    }

    void setTimeout(long long t) {
        this->timeout = std::chrono::duration<long long, std::micro>(t);
    }

    void setCallback(std::function<bool()> clbk) {
        this->callback = clbk;
    }

    void start() {
        std::unique_lock<std::mutex> guard(mutex);
        if(state != State::NEW) {
            throw ::arrus::IllegalStateException(
                "Only new watchdog object can be started.");
        }
        state = State::STARTED;
        logger->log(LogSeverity::DEBUG, "Starting.");
        this->thread = std::thread(&Watchdog::process, this);
    }

    void process() {
        logger->log(LogSeverity::DEBUG, "Started.");
        std::unique_lock<std::mutex> guard(mutex);
        bool internalStart = false;
        while(this->state == STARTED) {
            if(!internalStart) {
                logger->log(LogSeverity::DEBUG, "Waiting for the start.");
                cvStart.wait(guard, [this] {return this->startIndicator;});
                this->startIndicator = false;
                this->responseIndicator = false;
                logger->log(LogSeverity::DEBUG, "After the start.");
            }
            if(this->state == STOPPED) {
                break;
            }
            bool status = cvResponse.wait_for(guard, timeout, [this] {
                return this->responseIndicator;
            });

            if(this->state == STOPPED) {
                break;
            }
            if(!status) {
                logger->log(LogSeverity::DEBUG, "Timeout.");
                bool res = callback();
                internalStart = true;
                if(!res) {
                    // Queues closed.
                    this->state = STOPPED;
                    break;
                }
            }
            else {
                logger->log(LogSeverity::DEBUG, "No timeout.");
                internalStart = false;
            }
        }
        logger->log(LogSeverity::DEBUG, "Stopped.");
    }

    void stop() {
        {
            std::unique_lock<std::mutex> guard(mutex);
            logger->log(LogSeverity::DEBUG, "Stopping.");
            state = State::STOPPED;
        }
        cvStart.notify_all();
        cvResponse.notify_all();
        logger->log(LogSeverity::DEBUG, "Waiting to stop.");
        this->thread.join();
        logger->log(LogSeverity::DEBUG, "Stopped.");
    }

    void notifyStart() {
        {
            std::unique_lock<std::mutex> guard(mutex);
            logger->log(LogSeverity::DEBUG, "Start notification.");
            startIndicator = true;
        }
        cvStart.notify_one();
    }

    void notifyResponse() {
        {
            std::unique_lock<std::mutex> guard(mutex);
            logger->log(LogSeverity::DEBUG, "Notify response.");
            responseIndicator = true;
        }
        cvResponse.notify_one();
    }

private:

    enum State {
        NEW, STARTED, STOPPED
    };

    Logger::Handle logger;
    // Number of microseconds to wait for the device.
    std::chrono::duration<long long, std::micro> timeout;
    std::function<bool()> callback;
    std::mutex mutex;
    std::condition_variable cvStart;
    std::condition_variable cvResponse;
    State state{NEW};
    std::thread thread;
    bool responseIndicator{false};
    bool startIndicator{false};
};

}

#endif //ARRUS_CORE_DEVICES_US4R_WATCHDOG_H
