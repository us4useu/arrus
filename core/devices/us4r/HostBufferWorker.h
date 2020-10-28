#ifndef ARRUS_CORE_DEVICES_US4R_HOSTBUFFERWORKER_H
#define ARRUS_CORE_DEVICES_US4R_HOSTBUFFERWORKER_H

#include <utility>

#include "RxBuffer.h"
#include "Us4RHostBuffer.h"
#include "arrus/core/common/logging.h"

namespace arrus::devices {

class HostBufferWorker {
public:
    HostBufferWorker(std::shared_ptr<RxBuffer> inputBuffer,
                     std::shared_ptr<Us4RHostBuffer> outputBuffer,
                     std::vector<std::vector<DataTransfer>> transfers)
        : logger{getLoggerFactory()->getLogger()},
          inputBuffer(std::move(inputBuffer)),
          outputBuffer(std::move(outputBuffer)),
          transfers(std::move(transfers)) {

        INIT_ARRUS_DEVICE_LOGGER(logger, "HostBufferRunner");
    }

    void start() {
        std::unique_lock<std::mutex> guard(mutex);
        if(state == State::STARTED) {
            throw IllegalArgumentException("Worker already started.");
        }
        state = State::STARTED;
        logger->log(LogSeverity::DEBUG, "Starting host buffer runner.");
        this->processingThread = std::thread(&HostBufferWorker::process, this);
    }

    void process() {
        while(this->state == State::STARTED){
            logger->log(LogSeverity::DEBUG, "Waiting for queue element.");
            auto idx = inputBuffer->tail();
            if(idx < 0) {
                this->state = State::STOPPED;
                break;
            }
            auto &ts = transfers[idx];

            logger->log(LogSeverity::DEBUG, "Pushing data to the output buffer.");
            bool pushResult = outputBuffer->push([&ts, idx] (int16* dstAddress) {
                for(auto &t : ts) {
                    t.getTransferFunc()((uint8_t*)dstAddress);
                }
            });
            logger->log(LogSeverity::DEBUG, "Data transferred.");
            if(!pushResult) {
                this->state = State::STOPPED;
                break;
            }
            logger->log(LogSeverity::DEBUG, "Releasing buffer tail.");
            bool releaseTail = inputBuffer->releaseTail();
            logger->log(LogSeverity::DEBUG, "Released buffer tail.");
            if(!releaseTail) {
                this->state = State::STOPPED;
                break;
            }
        }
    }

private:
    enum class State{STARTED, STOPPED};

    Logger::Handle logger;
    std::thread processingThread;
    std::mutex mutex;
    std::shared_ptr<RxBuffer> inputBuffer;
    std::shared_ptr<Us4RHostBuffer> outputBuffer;
    // Element -> us4oem -> transfer
    std::vector<std::vector<DataTransfer>> transfers;
    State state{State::STOPPED};
};

}

#endif //ARRUS_CORE_DEVICES_US4R_HOSTBUFFERWORKER_H
