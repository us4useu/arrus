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
            logger->log(LogSeverity::DEBUG, "Waiting for rx.");
            auto idx = inputBuffer->tail();
            if(idx < 0) {
                this->state = State::STOPPED;
                break;
            }
            auto &ts = transfers[idx];

            logger->log(LogSeverity::DEBUG, ::arrus::format("Push rx {}.", idx));
            bool pushResult = outputBuffer->push([&ts, idx] (int16* dstAddress) {
                size_t offset = 0;
                for(auto &t : ts) {
                    t.getTransferFunc()((uint8_t*)dstAddress + offset);
                    offset += t.getSize();
                }
            });
            logger->log(LogSeverity::DEBUG, ::arrus::format("Push rx finished {}.", idx));
            if(!pushResult) {
                this->state = State::STOPPED;
                break;
            }
            logger->log(LogSeverity::DEBUG, ::arrus::format("Releasing rx {}.", idx));
            bool releaseTail = inputBuffer->releaseTail();
            logger->log(LogSeverity::DEBUG, ::arrus::format("Released rx {}.", idx));
            if(!releaseTail) {
                this->state = State::STOPPED;
                break;
            }
        }
        logger->log(LogSeverity::DEBUG, "Host buffer finished all work.");
    }

    void join() {
        this->processingThread.join();
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
};

}

#endif //ARRUS_CORE_DEVICES_US4R_HOSTBUFFERWORKER_H
