#ifndef ARRUS_CORE_API_DEVICES_US4R_IO_IOADDRESS_H
#define ARRUS_CORE_API_DEVICES_US4R_IO_IOADDRESS_H

#include <unordered_set>
#include <utility>

#include "arrus/core/api/devices/DeviceId.h"

namespace arrus::devices::us4r {

/**
 * Us4R IO address.
 *
 * Us4R IO address is a pair (us4oem, IO pin number), i.e. to address IO uniquely
 * within an us4R, we need to specify us4OEM ordinal number and pin number
 * of the IO on this us4OEM.
 */
class IOAddress {
public:
    using IO = uint8_t;
    IOAddress(Ordinal us4Oem, uint8_t io) : us4oem(us4Oem), io(io) {}

    /**
     * Returns us4OEM number of the address.
     */
    Ordinal getUs4OEM() const { return us4oem; }

    /**
     * Returns IO number, which refers to the us4OEM of this address.
     */
    uint8_t getIO() const { return io; }

    bool operator==(const IOAddress &rhs) const { return us4oem == rhs.us4oem && io == rhs.io; }
    bool operator!=(const IOAddress &rhs) const { return !(rhs == *this); }

private:
    Ordinal us4oem;
    IO io;
};

struct IOAddressHasher {
    std::size_t operator()(const IOAddress &addr) const {
        size_t result = std::hash<Ordinal>{}(addr.getUs4OEM());
        result ^= std::hash<IOAddress::IO>{}(addr.getIO()) + 0x9e3779b9 + (result << 6) + (result >> 2);
        return result;
    }
};

/**
 * A set of IO addresses.
 */
class IOAddressSet {

public:
    /** Creates empty set **/
    IOAddressSet() = default;

    explicit IOAddressSet(const std::vector<IOAddress> &addrVector)
        : addresses(std::begin(addrVector), std::end(addrVector)) {}
    size_t size() const { return addresses.size(); }
    bool contains(const IOAddress &addr) { return addresses.find(addr) != std::end(addresses); }

    auto begin() const {
        return std::begin(addresses);
    }

    auto end() const {
        return std::end(addresses);
    }

private:
    std::unordered_set<IOAddress, IOAddressHasher> addresses;
};

}// namespace arrus::devices::us4r

#endif//ARRUS_CORE_API_DEVICES_US4R_IO_IOADDRESS_H
