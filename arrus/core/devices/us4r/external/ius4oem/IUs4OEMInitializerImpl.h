#ifndef ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_IUS4OEMINITIALIZERIMPL_H
#define ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_IUS4OEMINITIALIZERIMPL_H

#include <algorithm>
#include <execution>
#include <mutex>
#include <iostream>

#include "IUs4OEMInitializer.h"

namespace arrus::devices {

class IUs4OEMInitializerImpl : public IUs4OEMInitializer {
public:

    void initModules(std::vector<IUs4OEMHandle> &ius4oems) override {
        // Reorder us4oems according to ids (us4oem with the lowest id is the
        // first one, with the highest id - the last one).
        // TODO(pjarosik) make the below sorting exception safe
        // (currently will std::terminate on an exception).
        std::sort(std::begin(ius4oems), std::end(ius4oems),
                  [](const IUs4OEMHandle &x, const IUs4OEMHandle &y) {
                      return x->GetID() < y->GetID();
                  });

        initializeUs4oems(ius4oems, 1);
        // Perform successive initialization levels.
        for(int level = 2; level <= 4; level++) {
            ius4oems[0]->Synchronize();
            initializeUs4oems(ius4oems, level);
        }
        // Us4OEMs are initialized here.
    }
private:
    void initializeUs4oems(std::vector<IUs4OEMHandle> &ius4oems, int level) {
        std::vector<std::exception> exceptions;
        std::mutex exceptions_mutex;

        std::for_each(
            std::execution::par_unseq, std::begin(ius4oems), std::end(ius4oems),
            [&](IUs4OEMHandle &u) {
                try {
                    u->Initialize(level);
                }
                catch(const std::exception &e) {
                    std::lock_guard guard(exceptions_mutex);
                    exceptions.push_back(e);
                }
                catch(...) {
                    std::lock_guard guard(exceptions_mutex);
                    exceptions.push_back(std::runtime_error("Unknown exception during Us4OEM initialization."));
                }
            }
        );
        if(!exceptions.empty()) {
            // TODO create a complete list of error messages
            // now throw any of the exception
            throw exceptions.front();
        }
    }
};

}

#endif //ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_IUS4OEMINITIALIZERIMPL_H
