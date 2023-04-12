#ifndef ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_IUS4OEMINITIALIZERIMPL_H
#define ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_IUS4OEMINITIALIZERIMPL_H

#include <algorithm>
#include <mutex>
#include <iostream>
#include <future>

#include "IUs4OEMInitializer.h"

namespace arrus::devices {

class IUs4OEMInitializerImpl : public IUs4OEMInitializer {
public:

    void sortModulesById(std::vector<IUs4OEMHandle> &ius4oems) override {
        // Sort us4oems according to ids (us4oem with the lowest id is the
        // first one, with the highest id - the last one).
        // TODO(pjarosik) make the below sorting exception safe
        // (currently will std::terminate on an exception).
        std::sort(std::begin(ius4oems), std::end(ius4oems),
                  [](const IUs4OEMHandle &x, const IUs4OEMHandle &y) {
                      return x->GetID() < y->GetID();
                  });
    }

    void initModules(std::vector<IUs4OEMHandle> &ius4oems, const std::vector<Us4OEMSettings> &) override {// us4oemCfgs) override {
        std::sort(std::begin(ius4oems), std::end(ius4oems),
                  [](const IUs4OEMHandle &x, const IUs4OEMHandle &y) {
                      return x->GetID() < y->GetID();
                  });
        // Set TX frequency range.
        // for(size_t i = 0; i < us4oemCfgs.size(); ++i) {
        //    ius4oems[i]->SetTxFrequencyRange(us4oemCfgs[i].getTxFrequencyRange());
        //}
        initializeUs4oems(ius4oems, 1);
        // Perform successive initialization levels.
        for(int level = 2; level <= 4; level++) {
            ius4oems[0]->Synchronize();
            initializeUs4oems(ius4oems, level);
        }
        // Us4OEMs are initialized here.
    }
private:

    void initializeUs4oems(const std::vector<IUs4OEMHandle> &ius4oems, int level) {
        std::vector<std::future<void>> results;

        for(const IUs4OEMHandle &ius4oem : ius4oems) {
            std::future<void> result = std::async(std::launch::async, [&ius4oem, level]() {ius4oem->Initialize(level);});
            results.push_back(std::move(result));
        }
        for(auto &result: results) {
            result.wait();
            result.get(); // wait and throw exception if necessary.
        }
    }
};

}

#endif //ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_IUS4OEMINITIALIZERIMPL_H
