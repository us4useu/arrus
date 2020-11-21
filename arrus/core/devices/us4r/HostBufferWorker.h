#ifndef ARRUS_CORE_DEVICES_US4R_HOSTBUFFERWORKER_H
#define ARRUS_CORE_DEVICES_US4R_HOSTBUFFERWORKER_H

#include <utility>

#include "RxBuffer.h"
#include "Us4RHostBuffer.h"
#include "arrus/core/common/logging.h"

namespace arrus::devices {

class HostBufferWorker {
public:
    HostBufferWorker(std::shared_ptr<Us4RHostBuffer> outputBuffer,
                     std::vector<std::vector<DataTransfer>> transfers,
                     long long priTimeout,
                     std::function<void()> syncFunc,
                     std::function<void()> startFunc)
        : logger{getLoggerFactory()->getLogger()},
          outputBuffer(std::move(outputBuffer)),
          transfers(std::move(transfers)),
          syncFunc(std::move(syncFunc)),
          startFunc(std::move(startFunc)),
          priTimeout(priTimeout)
    {
        INIT_ARRUS_DEVICE_LOGGER(logger, "HostBufferRunner");
    }

    void start() {
        std::unique_lock<std::mutex> guard(mutex);
        if(state != State::NEW) {
            throw ::arrus::IllegalStateException(
                "Only new host buffer worker threads can be started.");
        }
        state = State::STARTED;
        logger->log(LogSeverity::DEBUG, "Starting worker.");
        this->processingThread = std::thread(&HostBufferWorker::process, this);
    }

    void stop() {
        {
            std::unique_lock<std::mutex> guard(mutex);
            logger->log(LogSeverity::DEBUG, "Stopping worker.");
            state = State::STOPPED;
        }
        logger->log(LogSeverity::DEBUG, "Waiting for worker to stop.");
        this->processingThread.join();
        logger->log(LogSeverity::DEBUG, "Worker stopped.");
    }

    void process() {
        try {
            startFunc();
            std::this_thread::sleep_for(std::chrono::microseconds(priTimeout));
            while(this->state == State::STARTED) {
                syncFunc();
                std::this_thread::sleep_for(std::chrono::microseconds(priTimeout));
                auto &ts = transfers[0];
                bool pushResult = outputBuffer->push([&ts] (int16* dstAddress) {
                    size_t offset = 0;
                    for(auto &t : ts) {
                        t.getTransferFunc()((uint8_t*)dstAddress + offset);
                        offset += t.getSize();
                    }
                });
                if(!pushResult) {
                    this->state = State::STOPPED;
                    break;
                }
            }
            logger->log(LogSeverity::DEBUG, "Host buffer finished all work.");
        } catch(const std::exception &e) {
            std::cerr << e.what() << std::endl;
        } catch(...) {
            std::cerr << "Unhandled exception" << std::endl;
        }

    }
private:
    enum class State{NEW, STARTED, STOPPED};

    Logger::Handle logger;
    std::thread processingThread;
    std::mutex mutex;
    std::shared_ptr<RxBuffer> inputBuffer;
    std::shared_ptr<Us4RHostBuffer> outputBuffer;
    // Element -> us4oem -> transfer
    std::vector<std::vector<DataTransfer>> transfers;
    State state{State::NEW};
    long long priTimeout;
    std::function<void()> syncFunc;
    std::function<void()> startFunc;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_HOSTBUFFERWORKER_H
