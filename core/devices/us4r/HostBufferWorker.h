#ifndef ARRUS_CORE_DEVICES_US4R_HOSTBUFFERWORKER_H
#define ARRUS_CORE_DEVICES_US4R_HOSTBUFFERWORKER_H

#include <utility>

#include "RxBuffer.h"
#include "Us4RHostBuffer.h"

namespace arrus::devices {

class HostBufferWorker {
public:
    HostBufferWorker(std::shared_ptr<RxBuffer> inputBuffer,
                     std::shared_ptr<Us4RHostBuffer> outputBuffer,
                     std::vector<std::vector<DataTransfer>> transfers)
        : inputBuffer(std::move(inputBuffer)),
          outputBuffer(std::move(outputBuffer)),
          transfers(std::move(transfers)) {}

    void start() {
        // TODO create a thread and start
    }

    void process() {
        while(true) {
            auto idx = inputBuffer->tail();
            auto &ts = transfers[idx];

            outputBuffer->push([&ts, idx] (int16* dstAddress) {
                for(auto &t : ts) {
                    t.getTransferFunc()((uint8_t*)dstAddress);
                }
            });
            inputBuffer->releaseTail();
        }
    }

private:
    std::thread processingThread;
    std::shared_ptr<RxBuffer> inputBuffer;
    std::shared_ptr<Us4RHostBuffer> outputBuffer;
    // Element -> us4oem -> transfer
    std::vector<std::vector<DataTransfer>> transfers;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_HOSTBUFFERWORKER_H
