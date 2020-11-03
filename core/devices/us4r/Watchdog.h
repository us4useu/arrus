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

    Watchdog() = default;

    Watchdog(long long int timeout, std::function<void()> callback)
        : logger{getLoggerFactory()->getLogger()},
        timeout(timeout), callback(std::move(callback)) {
        INIT_ARRUS_DEVICE_LOGGER(logger, "Watchdog");
    }

    void setTimeout(long long timeout) {
        this->timeout = std::chrono::duration<long long, std::micro>(timeout);
    }

    void setCallback(std::function<void()> clbk) {
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
        // for start
        while(this->state == STARTED) {
            cvStart.wait(guard);
            if(this->state == STOPPED) {
                break;
            }

            auto status = cvResponse.wait_for(guard, timeout);
            if(this->state == STOPPED) {
                break;
            }
            if(status == std::cv_status::timeout) {
                logger->log(LogSeverity::DEBUG, "Timeout.");
                callback();
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
        cvStart.notify_one();
    }

    void notifyResponse() {
        cvResponse.notify_one();
    }
private:

    enum State {NEW, STARTED, STOPPED};

    Logger::Handle logger;
    // Number of microseconds to wait for the device.
    std::chrono::duration<long long, std::micro> timeout;
    std::function<void()> callback;
    std::mutex mutex;
    std::condition_variable cvStart;
    std::condition_variable cvResponse;
    State state{NEW};
    std::thread thread;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_WATCHDOG_H
