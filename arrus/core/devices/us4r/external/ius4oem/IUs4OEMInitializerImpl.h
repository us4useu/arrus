#ifndef ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_IUS4OEMINITIALIZERIMPL_H
#define ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_IUS4OEMINITIALIZERIMPL_H

#include "IUs4OEMInitializer.h"

namespace arrus::devices {

class IUs4OEMInitializerImpl : public IUs4OEMInitializer {
public:

    void initModules(std::vector<IUs4OEMHandle> &ius4oems) override {
        // Reorder us4oems according to ids (us4oem with the lowest id is the
        // first one, with the highest id - the last one).
        std::vector<std::pair<unsigned int, Ius4OEMRawHandle>> sortedIus4oems;

        for(auto &ius4oem: ius4oems) {
            sortedIus4oems.emplace_back(ius4oem->GetID(), ius4oem.get());
        }

        std::sort(std::begin(sortedIus4oems), std::end(sortedIus4oems),
                  [](const auto &x, const auto &y) {return x.first < y.first;});

        for(auto &[id, u] : sortedIus4oems) {
            u->Initialize(1);
        }
        // Perform successive initialization levels.
        for(int level = 2; level <= 4; level++) {
            sortedIus4oems[0].second->Synchronize();
            for(auto &[id, u] : sortedIus4oems) {
                u->Initialize(level);
            }
        }
        // Us4OEMs are initialized here.
    }

};

}

#endif //ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_IUS4OEMINITIALIZERIMPL_H
