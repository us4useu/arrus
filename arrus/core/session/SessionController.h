#ifndef ARRUS_ARRUS_CORE_SESSION_SESSIONCONTROLLER_H
#define ARRUS_ARRUS_CORE_SESSION_SESSIONCONTROLLER_H

#include <thread>

#include "arrus/common/format.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/kernels/Kernel.h"
#include "BlockingQueue.h"

namespace arrus::session {

class SessionController {
public:

    SessionController(): logger{getLoggerFactory()->getLogger()} {
        INIT_ARRUS_DEVICE_LOGGER(logger, "SessionController");
        // Initialize kernels that should be always available: SetHVVoltage, etc.
    }

    void start() {
        this->isRunning = true;
        this->thread = std::thread(&SessionController::process, this);
    }

    void stop() {
        this->isRunning = false;
    }

    void run(const ::arrus::ops::Op::SharedHandle& op) {
        queue.push(op);
    }

private:

    void process() {
        try {
            ::arrus::ops::Op::SharedHandle op;
            while(isRunning) {
                bool result = queue.pop(op);
                if(!result) {
                    logger->log(LogSeverity::DEBUG, "Op queue shutdown");
                    return;
                }
                auto &kernel = kernels.at(op->getTypeId());
                kernel.process(op);
            }
        }
        catch(const std::exception &e) {
            logger->log(LogSeverity::DEBUG, ::arrus::format("Stopping, error: {}", e.what()));
        }
    }
    Logger::Handle logger;
    bool isRunning{false};
    std::thread thread;
    BlockingQueue<::arrus::ops::Op::SharedHandle> queue;
    // op type id -> kernel
};

}

#endif //ARRUS_ARRUS_CORE_SESSION_SESSIONCONTROLLER_H
