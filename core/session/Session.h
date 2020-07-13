#ifndef ARRUS_CORE_SESSION_SESSION_H
#define ARRUS_CORE_SESSION_SESSION_H

#include <memory>

namespace arrus {
	class Session {
	public:

	    using DeviceHandle = std::shared_ptr<Device>;

		Session();

		DeviceHandle getDevice();



	private:

	};
}


#endif //ARRUS_CORE_SESSION_SESSION_H
