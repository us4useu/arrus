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
                     std::vector<std::vector<DataTransfer>> transfers,
                     long long rxdmaTimeout)
        : logger{getLoggerFactory()->getLogger()},
          inputBuffer(std::move(inputBuffer)),
          outputBuffer(std::move(outputBuffer)),
          transfers(std::move(transfers)),
          rxdmaTimeout(rxdmaTimeout) {
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
        int16_t i = 0;
        while(this->state == State::STARTED){
            logger->log(LogSeverity::DEBUG, "Waiting for rx.");
            auto idx = inputBuffer->tail(rxdmaTimeout);
            if(idx == -1) {
                this->state = State::STOPPED;
                break;
            } else if (idx == -2) {
                logger->log(LogSeverity::INFO, "A timeout while waiting for rx dma response.");
                // anyway, continue processing
                // TODO the rx buffer and the host buffer current positions may misalign here
                // move the position of the buffer
                idx = i;
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

            i = (i+1) % (inputBuffer->size());
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
    long long rxdmaTimeout;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_HOSTBUFFERWORKER_H
